"""HistoricalReplayEngine on synthetic recordings (design §7, §10.2 scenarios 13 to 19)."""

from decimal import Decimal

from execsim_helpers import BTC, META, Sealer, gz_member, level3, order, recording_lines, write_file

from toxicflow.execsim.config import SimConfig
from toxicflow.execsim.models import EndReason, Mechanism, OrderIntent, PlacementRequest, Side, SkipReason
from toxicflow.execsim.orders import OrderState, SimulatedOrder
from toxicflow.execsim.reader import KIND_META, RecordingReader, book_items
from toxicflow.execsim.replay import HistoricalReplayEngine
from toxicflow.execsim.strategies import JoinBestStrategy
from toxicflow.execsim.timeutil import MS, format_ns, parse_rfc3339_ns
from toxicflow.kraken.book import L3Book

D = Decimal
BUY, SELL = Side.BUY, Side.SELL
SYM = "BTC/USD"
T0 = parse_rfc3339_ns("2026-10-18T02:00:00Z")
BIDS = [order("B1", "100.0", "1.0"), order("B2", "100.0", "2.0")]
ASKS = [order("A1", "100.2", "3.0")]


def at(ms: float) -> int:
    return T0 + int(ms * MS)


class Tape:
    """A synthetic L3 recording: items with chosen exchange times (ms after T0) and receive times."""

    def __init__(self, depth=10):
        self.sealer = Sealer(BTC, depth)
        self.meta = {**META, "depth": depth}
        self.files: list[list] = [[]]

    def snapshot(self, ms, bids=BIDS, asks=ASKS, recv_ms=None):
        return self._add(ms, "snapshot", self.sealer.snapshot(bids=bids, asks=asks), recv_ms)

    def update(self, ms, bids=(), asks=(), corrupt=False, recv_ms=None):
        item = self.sealer.update(bids=bids, asks=asks)
        if corrupt:
            item["checksum"] += 1
        return self._add(ms, "update", item, recv_ms)

    def new_file(self):
        self.files.append([])

    def _add(self, ms, msg_type, item, recv_ms):
        item["timestamp"] = format_ns(at(ms))
        self.files[-1].append((at(ms if recv_ms is None else recv_ms), level3(msg_type, item)))
        return item

    def write(self, root):
        return [write_file(root, f"2026-10-18T{2 + i:02d}", gz_member(recording_lines(msgs, self.meta)))
                for i, msgs in enumerate(self.files)]


def req(ms, side=BUY):
    return PlacementRequest(f"{side.value}@{ms}", SYM, side, at(ms))


def engine(schedule, horizon_ms=1_000, latency_ms=0, strategies=None, **kw):
    cfg = SimConfig(symbols=(SYM,), order_qty={SYM: "0.5"}, horizon_ns=int(horizon_ms * MS),
                    latency_ns=int(latency_ms * MS), audit_invariants=True, **kw)
    return HistoricalReplayEngine(cfg, strategies or [JoinBestStrategy()], schedule)


def by_id(result):
    return {o.request.request_id: o for o in result.orders}


def skips(result):
    return {(s.request.request_id, s.reason) for s in result.skips}


def kinds(result):
    return [q.kind for q in result.quality]


# --- end to end ---------------------------------------------------------------------------


def test_end_to_end_placement_advancement_and_reaches(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])  # behind us
    tape.update(20, bids=[order("B1", "100.0", "0.6", "modify")])  # ahead shrinks
    tape.update(30, bids=[order("B2", "100.0", "2.0", "delete")])
    tape.update(40, bids=[order("B1", "100.0", "0.6", "delete")])  # we reach the front
    tape.update(50, bids=[order("B3", "100.0", "3.8", "modify")])
    tape.update(60, bids=[order("B3", "100.0", "3.2", "modify")])
    seen = []
    eng = engine([req(5)])
    eng.on_reach = lambda o, reach, ev: seen.append((o.order_id, ev.order_id))
    result = eng.run(tape.write(tmp_path))

    (o,) = result.orders
    assert (o.order_id, o.price, o.qty, o.inserted_ts, o.expiry_ts) == ("buy@5/join_best", D("100.0"), D("0.5"),
                                                                     at(5), at(1_005))
    assert (o.queue_ahead_qty0, o.queue_ahead_orders0, o.touch_qty0, o.level_rank0) == (D("3.0"), 2, D("3.0"), 1)
    assert o.tracker.at_front and o.tracker.advanced_qty == D("3.0") and o.tracker.ahead_qty == 0
    assert [(r.reach.volume, r.reach.mechanism, r.reach.item_ts_ns, r.ahead_qty) for r in result.reaches] == [
        (D("0.2"), Mechanism.LEVEL, at(50), 0), (D("0.6"), Mechanism.LEVEL, at(60), 0)]
    assert o.fills == []  # reaches are not converted into fills yet
    assert seen == [("buy@5/join_best", "B3")] * 2
    assert (o.end_reason, o.end_ts) == (EndReason.CENSORED_END_OF_DATA, at(60))  # data ended before the horizon
    c = result.counters[SYM]
    assert (c["snapshots"], c["checksum_ok"], c["checksum_fail"], c["audit_fail"]) == (1, 7, 0, 0)
    # the reach ref points back to the raw line through the reader's seq
    recs = list(RecordingReader(files=tape.write(tmp_path / "again")))
    first = result.reaches[0].reach.ref
    assert recs[first[0]].line_no == 7 and first[1] == 0


# --- clock and strict-before -------------------------------------------------------------------


def test_decision_sees_only_items_before_its_time(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B5", "100.1", "1.0", "add")])  # new best bid at exactly t=10
    tape.update(20, asks=[order("A1", "100.2", "1.0", "modify")])
    result = engine([req(5), req(10), req(15)]).run(tape.write(tmp_path))
    orders = by_id(result)
    assert orders["buy@5"].price == D("100.0") and orders["buy@10"].price == D("100.0")  # t=10 item unseen
    assert orders["buy@15"].price == D("100.1")
    for view in result.views.values():
        assert parse_rfc3339_ns(view.book_timestamp) < view.ts_ns
    # the order decided at 10 is inserted before the t=10 item: B5 is a better price, not ahead of us
    assert list(orders["buy@10"].tracker.ahead_ids) == ["B1", "B2"]


def test_equal_timestamps_go_after_insertion(tmp_path):  # scenario 17, §7.1 example
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "0.5", "modify")])  # i1
    tape.update(20, bids=[order("B9", "100.0", "1.0", "add")])  # i2, same time as the request
    tape.update(20, bids=[order("B1", "100.0", "0.3", "modify")])  # i3, same time
    tape.update(30, asks=[order("A1", "100.2", "2.0", "modify")])  # i4
    result = engine([req(20)]).run(tape.write(tmp_path))
    view = result.views["buy@20"]
    assert view.bids[0].qty == D("2.5")  # holds i1 only
    t = result.orders[0].tracker
    assert list(t.ahead_ids.items()) == [("B1", D("0.3")), ("B2", D("2.0"))]  # i3 counted, B9 behind
    assert result.orders[0].inserted_ts == at(20)


def test_several_decisions_in_one_interval(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])
    result = engine([req(6), req(2, SELL), req(2), req(4)]).run(tape.write(tmp_path))
    # by time; equal times keep schedule order (R1, R7)
    assert [o.request.request_id for o in result.orders] == ["sell@2", "buy@2", "buy@4", "buy@6"]
    buys = [o for o in result.orders if o.side is BUY]
    assert [o.inserted_ts for o in buys] == [at(2), at(4), at(6)]
    assert all(list(o.tracker.ahead_ids) == ["B2"] for o in buys)  # each saw the t=10 delete
    views = [result.views[r] for r in ("buy@2", "buy@4", "buy@6")]
    assert [v.ts_ns for v in views] == [at(2), at(4), at(6)] and len({(v.bids, v.asks) for v in views}) == 1
    assert result.views["sell@2"] is result.views["buy@2"]  # same T: one shared view (R7)
    assert result.orders[0].price == D("100.2")


def test_all_strategies_share_one_view(tmp_path):
    seen = []

    class Spy:
        name = "spy"

        def decide(self, request, view):
            seen.append(view)
            return JoinBestStrategy().decide(request, view)

    class Spy2(Spy):
        name = "spy2"

    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    result = engine([req(5)], strategies=[Spy(), Spy2()]).run(tape.write(tmp_path))
    assert seen[0] is seen[1] is result.views["buy@5"]
    assert {o.order_id for o in result.orders} == {"buy@5/spy", "buy@5/spy2"}


def test_expiry_boundary(tmp_path):  # scenario 19
    tape = Tape()
    tape.snapshot(0)
    tape.update(104, bids=[order("B2", "100.0", "1.5", "modify")])  # before expiry: counts
    tape.update(105, bids=[order("B1", "100.0", "1.0", "delete")])  # exactly at expiry: does not
    tape.update(200)
    result = engine([req(5)], horizon_ms=100).run(tape.write(tmp_path))
    (o,) = result.orders
    assert (o.end_reason, o.end_ts) == (EndReason.HORIZON, at(105)) and not o.censored
    assert list(o.tracker.ahead_ids.items()) == [("B1", D("1.0")), ("B2", D("1.5"))]


def test_timestamp_regression_is_guarded(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(30)
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete")])  # out of order: clock stays at 30
    tape.update(40)
    result = engine([req(25)]).run(tape.write(tmp_path))
    assert "TS_REGRESSION" in kinds(result) and result.counters[SYM]["ts_regressions"] == 1
    assert list(result.orders[0].tracker.ahead_ids) == ["B2"]  # stream order still applies the delete


# --- skips ---------------------------------------------------------------------------------------


def test_skip_reasons(tmp_path):
    tape = Tape()
    tape.update(0)  # from an old subscription: before any snapshot
    tape.snapshot(10, bids=[], asks=ASKS)  # one-sided: no bids
    tape.update(20, bids=[order("B1", "100.0", "1.0", "add")])
    tape.update(30)
    result = engine([req(-5), req(5), req(15), req(25), req(99)]).run(tape.write(tmp_path))
    assert skips(result) == {("buy@-5", SkipReason.UNSYNCED), ("buy@5", SkipReason.UNSYNCED),
                             ("buy@15", SkipReason.NO_TOUCH), ("buy@99", SkipReason.NO_DATA)}
    assert [o.request.request_id for o in result.orders] == ["buy@25"]
    assert result.counters[SYM]["skipped_unsynced"] == 1


def test_warmup_after_sync(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(100)
    result = engine([req(30), req(60)], warmup_ns=50 * MS).run(tape.write(tmp_path))
    assert skips(result) == {("buy@30", SkipReason.WARMUP)} and len(result.orders) == 1


def test_crossed_at_insert_with_latency(tmp_path):  # scenario 18
    tape = Tape()
    tape.snapshot(0)
    tape.update(20, asks=[order("A2", "100.0", "1.0", "add")])  # the ask comes down onto our price
    tape.update(100)
    result = engine([req(5), req(5, SELL)], latency_ms=50).run(tape.write(tmp_path))
    orders = by_id(result)
    assert orders["buy@5"].price == D("100.0")  # priced from the decision view
    assert orders["buy@5"].skip_reason is SkipReason.CROSSED_AT_INSERT and orders["buy@5"].tracker is None
    sell = orders["sell@5"]
    assert sell.price == D("100.2") and sell.inserted_ts == at(55) and sell.decided_ts == at(5)
    assert list(sell.tracker.ahead_ids) == ["A1"]  # queue read at insertion (A2 is better-priced)


def test_pending_insert_at_end_of_data(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    result = engine([req(5)], latency_ms=50).run(tape.write(tmp_path))
    (o,) = result.orders
    assert o.state is OrderState.ENDED and o.skip_reason is SkipReason.NO_DATA


# --- discontinuities ----------------------------------------------------------------------------


def test_desync_censors_then_snapshot_restores(tmp_path):  # scenario 13
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B2", "100.0", "1.0", "modify")])
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete")], corrupt=True)
    tape.update(30, bids=[order("B9", "100.0", "1.0", "add")])  # ignored: unsynced
    tape.snapshot(40)
    tape.update(50, bids=[order("B1", "100.0", "0.4", "modify")])
    result = engine([req(5), req(25), req(45)]).run(tape.write(tmp_path))
    orders = by_id(result)
    first = orders["buy@5"]
    assert (first.end_reason, first.end_ts) == (EndReason.CENSORED_DESYNC, at(10))
    assert list(first.tracker.ahead_ids.items()) == [("B1", D("1.0")), ("B2", D("1.0"))]  # failed item unused
    assert skips(result) == {("buy@25", SkipReason.UNSYNCED)}
    assert orders["buy@45"].state is OrderState.ENDED and orders["buy@45"].tracker.ahead_qty == D("2.4")
    assert kinds(result).count("DESYNC") == 1 and "RESNAPSHOT" in kinds(result)
    c = result.counters[SYM]
    assert (c["checksum_fail"], c["skipped_unsynced"]) == (1, 1)


def test_resnapshot_censors_live_orders_and_keeps_reaches(tmp_path):  # scenario 14
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                          order("B3", "100.0", "1.0", "add")])
    tape.update(20, bids=[order("B3", "100.0", "0.5", "modify")])  # reach
    tape.snapshot(50)
    tape.update(60)
    result = engine([req(5), req(30), req(55)]).run(tape.write(tmp_path))
    orders = by_id(result)
    assert (orders["buy@5"].end_reason, orders["buy@5"].end_ts) == (EndReason.CENSORED_RESNAPSHOT, at(20))
    assert [r.reach.volume for r in result.reaches] == [D("0.5")]
    assert skips(result) == {("buy@30", SkipReason.UNSYNCED)}  # decided inside the hole before the snapshot
    assert orders["buy@55"].end_reason is EndReason.CENSORED_END_OF_DATA


def test_receive_gap_censors_and_needs_one_valid_item(tmp_path):  # scenario 16
    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    tape.update(6_100, recv_ms=6_100)  # 6 s of silence: the reader yields a gap marker first
    tape.update(6_200)
    result = engine([req(5), req(3_000), req(6_150)], horizon_ms=10_000).run(tape.write(tmp_path))
    orders = by_id(result)
    assert (orders["buy@5"].end_reason, orders["buy@5"].end_ts) == (EndReason.CENSORED_RECV_GAP, at(10))
    assert skips(result) == {("buy@3000", SkipReason.DATA_GAP)}
    assert orders["buy@6150"].inserted_ts == at(6_150)
    assert "RECV_GAP" in kinds(result)


def test_file_boundary_restarts_the_session(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    tape.new_file()
    tape.update(20, bids=[order("B1", "100.0", "0.5", "modify")])  # old subscription, before the snapshot
    tape.snapshot(30)
    tape.update(40)
    result = engine([req(5), req(25), req(35)]).run(tape.write(tmp_path))
    orders = by_id(result)
    assert (orders["buy@5"].end_reason, orders["buy@5"].end_ts) == (EndReason.CENSORED_SESSION_RESTART, at(10))
    assert skips(result) == {("buy@25", SkipReason.UNSYNCED)}
    assert orders["buy@35"].end_reason is EndReason.CENSORED_END_OF_DATA
    assert result.counters[SYM]["skipped_unsynced"] == 1 and kinds(result).count("SESSION_RESTART") == 2


def test_truncated_file_censors(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    (path,) = tape.write(tmp_path)
    path.write_bytes(path.read_bytes()[:-8])  # cut the gzip trailer
    result = engine([req(5)]).run([path])
    (o,) = result.orders
    assert (o.end_reason, o.end_ts) == (EndReason.CENSORED_TRUNCATED, at(10))
    assert kinds(result) == ["SESSION_RESTART", "SNAPSHOT_FIRST", "TRUNCATED", "END_OF_DATA"]


def test_depth_exit_ends_order(tmp_path):  # scenario 15
    tape = Tape(depth=2)
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=[order("A1", "101.0", "1.0")])
    tape.update(10, bids=[order("B6", "100.1", "1.0", "add")])  # rank 2: still visible
    tape.update(20, bids=[order("B7", "100.2", "1.0", "add")])  # 100.0 falls out of depth 2
    tape.update(30)
    result = engine([req(5)]).run(tape.write(tmp_path))
    (o,) = result.orders
    assert (o.end_reason, o.end_ts, o.max_level_rank) == (EndReason.OUT_OF_DEPTH, at(20), 2)


# --- the historical book ------------------------------------------------------------------------------


def test_replay_with_orders_keeps_the_historical_book(tmp_path):  # invariant I1
    tape = Tape()
    tape.snapshot(0)
    for i in range(1, 30):
        bids = [order(f"N{i}", "100.0", "1.0", "add")]
        if i % 3 == 0:
            bids.append(order(f"N{i - 1}", "100.0", "0.5", "modify"))
        if i % 5 == 0:
            bids.append(order(f"N{i - 2}", "100.0", "1.0", "delete"))
        tape.update(i * 10, bids=bids, asks=[order("A1", "100.2", str(3 + i), "modify")])
    files = tape.write(tmp_path)
    eng = engine([req(ms, side) for ms in range(5, 300, 20) for side in (BUY, SELL)], horizon_ms=50)
    plain = None
    for rec in RecordingReader(files=files):
        eng.process_record(rec)
        if rec.kind == KIND_META:
            plain = L3Book(BTC, rec.payload["depth"])
        for msg_type, item in book_items(rec):
            plain.apply(msg_type, item)
            sim = eng.state[SYM].book
            assert (sim.bids, sim.asks, sim.orders, sim.checksum()) == (plain.bids, plain.asks, plain.orders,
                                                                       plain.checksum())
    result = eng.finish()
    assert len(result.orders) == 30 and all(o.state is OrderState.ENDED for o in result.orders)
    assert not any(o.order_id in eng.state[SYM].book.orders for o in result.orders)
    assert result.counters[SYM]["audit_fail"] == 0


def test_rerun_is_identical(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(20, bids=[order("B3", "100.0", "1.0", "add")])
    tape.update(30, bids=[order("B3", "100.0", "1.0", "delete")])
    files = tape.write(tmp_path)

    def summary():
        r = engine([req(5), req(5, SELL), req(15)]).run(files)
        return ([(o.order_id, o.end_reason, o.end_ts, dict(o.tracker.ahead_ids)) for o in r.orders],
                [(x.order_id, x.reach) for x in r.reaches], [(q.kind, q.ts_ns, q.seq) for q in r.quality])

    assert summary() == summary()


# --- checksum coverage ----------------------------------------------------------------------------


def test_order_beyond_checksummed_levels_is_not_covered(tmp_path):
    tape = Tape(depth=20)
    tape.snapshot(0, asks=[order("A1", "102.0", "1.0")])
    better = [order(f"X{i}", str(D("100.1") + D("0.1") * i), "1.0", "add") for i in range(10)]
    tape.update(10, bids=better[:9])  # our level becomes rank 10: still checksummed
    tape.update(20)
    tape.update(30, bids=better[9:])  # rank 11: visible (depth 20) but outside the checksum
    tape.update(40)
    files = tape.write(tmp_path)
    (short,) = engine([req(5)], horizon_ms=20).run(files).orders  # expires at 25, before rank 11
    (long,) = engine([req(5)], horizon_ms=100).run(files).orders  # lives through the item at 30
    assert short.price == long.price == D("100.0")
    assert (short.end_reason, short.max_level_rank, short.checksum_covered) == (EndReason.HORIZON, 10, True)
    assert (long.max_level_rank, long.checksum_covered) == (11, False)
    assert long.end_reason is EndReason.CENSORED_END_OF_DATA  # still visible: depth 20
    assert SimulatedOrder.new(req(1), "x", OrderIntent(BUY, D("1")), D("1"), 0).checksum_covered is None


def test_reach_coverage_follows_the_validated_post_item_book(tmp_path):
    tape = Tape(depth=20)
    tape.snapshot(0, bids=[order("B1", "100.0", "1.0")], asks=[order("A1", "102.0", "1.0")])
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])  # we are alone at 100.0
    # eleven asks arrive through our bid; after the item they rank 1..11 on the ask side
    tape.update(20, asks=[order(f"C{i}", str(D("99.0") + D("0.1") * i), "0.1", "add") for i in range(11)])
    tape.update(30)
    result = engine([req(5)]).run(tape.write(tmp_path))
    assert [r.reach.mechanism for r in result.reaches] == [Mechanism.CROSS_ARRIVAL] * 11
    assert [r.checksum_covered for r in result.reaches] == [True] * 10 + [False]
    assert result.orders[0].checksum_covered is True  # our own level stayed at rank 1


def test_failed_checksum_item_produces_no_reach(tmp_path):  # I7 with a would-be LEVEL candidate
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                          order("B3", "100.0", "1.0", "add")])
    tape.update(20, bids=[order("B3", "100.0", "0.5", "modify")], corrupt=True)
    tape.update(30, bids=[order("B3", "100.0", "0.2", "modify")])  # unsynced: ignored
    result = engine([req(5)]).run(tape.write(tmp_path))
    (o,) = result.orders
    assert result.reaches == [] and (o.end_reason, o.end_ts) == (EndReason.CENSORED_DESYNC, at(10))
    assert o.tracker.at_front and o.tracker.diag["events_after_invalid"] == 0  # nothing was even offered


# --- quality incidents ------------------------------------------------------------------------------


def test_records_of_one_break_share_an_incident(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10)
    tape.update(20, corrupt=True)  # incident 2: DESYNC ...
    tape.snapshot(30)  # ... and the RESNAPSHOT that repairs it
    tape.update(40)
    tape.update(6_100, recv_ms=6_100)  # incident 3: RECV_GAP, validated by this same item
    tape.update(12_200, recv_ms=12_200)  # incident 4: a second gap after validation is a new break
    tape.update(18_300, recv_ms=18_300, corrupt=True)  # incident 5: RECV_GAP then DESYNC before validation
    tape.snapshot(18_310, recv_ms=18_310)  # ... and its RESNAPSHOT
    tape.new_file()
    tape.update(18_320, recv_ms=18_320)  # incident 6: SESSION_RESTART ...
    tape.snapshot(18_330, recv_ms=18_330)  # ... and RESNAPSHOT
    result = engine([]).run(tape.write(tmp_path))
    breaks = [(q.kind, q.incident) for q in result.quality if q.incident is not None]
    assert breaks == [("SESSION_RESTART", 1), ("DESYNC", 2), ("RESNAPSHOT", 2), ("RECV_GAP", 3), ("RECV_GAP", 4),
                      ("RECV_GAP", 5), ("DESYNC", 5), ("RESNAPSHOT", 5), ("SESSION_RESTART", 6),
                      ("RESNAPSHOT", 6)]
    assert len({(q.symbol, q.incident) for q in result.quality if q.incident is not None}) == 6
    assert all(q.incident is None for q in result.quality if q.kind in ("SNAPSHOT_FIRST", "END_OF_DATA"))
