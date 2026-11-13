"""Trade evidence (trades.py) and trade-aware settlement and metrics (design §6.5.2 to §6.5.5, Phase 3).

Synthetic only: an L3 Tape plus a public-stream recording of trade prints for the same times.
Kraken stamps a trade with the L3 item time truncated to the microsecond, so `at(ms)` times
(whole microseconds) match exactly and `+ US` is one microsecond late.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from execsim_helpers import dumps, gz_member, order, write_file
from test_execsim_replay import BIDS, SYM, T0, Tape, at, engine, req

from toxicflow.execsim.execution import FillRecorder, FillStatus, settle
from toxicflow.execsim.metrics import analyze, format_report
from toxicflow.execsim.models import EndReason, Evidence, EvidencePart, FillCandidate, Mechanism, OrderIntent, \
    PlacementRequest, Side
from toxicflow.execsim.orders import SimulatedOrder
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy
from toxicflow.execsim.trades import MatchTolerance, TradeIndex, attribute_trades, is_plausible_execution

D = Decimal
BUY, SELL = Side.BUY, Side.SELL
US = 1_000
CONF, PLAUS, ABSENT = Evidence.TRADE_CONFIRMED, Evidence.TRADE_PLAUSIBLE, Evidence.TRADE_ABSENT
DEL, MOD, CROSS = Evidence.L3_DELETE_AMBIGUOUS, Evidence.L3_MODIFY_DOWN, Evidence.L3_CROSS_ARRIVAL
PUBLIC_META = {"stream": "public", "depth": 100, "pairs": {SYM: {"price_precision": 1, "qty_precision": 8}}}


def iso(ns: int, digits: int = 6) -> str:
    secs, frac = divmod(ns, 1_000_000_000)
    stamp = datetime.fromtimestamp(secs, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{stamp}.{frac // 10 ** (9 - digits):0{digits}d}Z"


class Public:
    """A public-stream recording: trade prints plus book items that mark exchange-time coverage."""

    def __init__(self, cover=(0, 3_000), next_id=1000):
        self.msgs: list[tuple[int, dict]] = []
        self.next_id = next_id
        for ms in cover:
            self.book(ms)

    def book(self, ms):
        self.msgs.append((at(ms), {"channel": "book", "type": "update", "data": [
            {"symbol": SYM, "bids": [], "asks": [], "checksum": 0, "timestamp": iso(at(ms))}]}))

    def trade(self, ts_ns, taker, price, qty, trade_id="auto", **extra):
        """One message of prints at one exchange time; `taker` is 'buy'/'sell' or anything else."""
        if trade_id == "auto":
            trade_id, self.next_id = self.next_id, self.next_id + 1
        t = {"symbol": SYM, "side": taker, "price": D(price), "qty": D(qty), "ord_type": "limit",
             "trade_id": trade_id, "timestamp": iso(ts_ns), **extra}
        if trade_id is None:
            del t["trade_id"]
        self.msgs.append((ts_ns, {"channel": "trade", "type": "update", "data": [t]}))
        return t

    def gap(self):
        self.msgs.append(("GAP", None))

    def write(self, root):
        lines, recv = [dumps({"recv_ns": 0, "meta": PUBLIC_META})], T0
        for ns, msg in self.msgs:
            if ns == "GAP":
                recv += 10_000_000_000  # beyond the 5 s receive-gap threshold
                continue
            recv += 1
            lines.append(f'{{"recv_ns":{recv},"msg":{dumps(msg)}}}')
        return [write_file(root, "2026-10-18T02", gz_member(lines), stream="public")]


def cfg_for(eng_cfg=None):
    return engine([]).cfg if eng_cfg is None else eng_cfg


def run(tmp_path, tape, public, tol=MatchTolerance(), schedule=(), strategies=None, attribution=True, **kw):
    """Attribution pass + replay with FillRecorder(attribution). Returns (result, attribution, index)."""
    l3 = tape.write(tmp_path / "l3")
    eng = engine(list(schedule), strategies=strategies, **kw)
    att = idx = None
    if attribution:
        idx = TradeIndex.from_files(public.write(tmp_path / "pub"), [SYM], tolerance=tol)
        att = attribute_trades(eng.cfg, idx, files=l3)
    eng.on_reach = FillRecorder(att)
    return eng.run(l3), att, idx


def labels(att, order_id):
    """[(evidence, qty, trade_ids)] for the attributed reductions of one historical order, in stream order."""
    return [[(p.evidence, p.qty, p.trade_ids) for p in r.parts]
            for _, r in sorted(att.by_ref.items()) if r.order_id == order_id]


def basic_tape():
    tape = Tape()
    tape.snapshot(0)  # bids B1 1.0, B2 2.0 at 100.0; ask A1 3.0 at 100.2
    return tape


# --- TradeIndex ------------------------------------------------------------------------------------------


def test_trade_index_parses_observed_fields_only(tmp_path):
    pub = Public()
    pub.trade(at(10), "sell", "100.0", "0.4")
    pub.trade(at(20), "buy", "100.2", "1.5")
    idx = TradeIndex.from_files(pub.write(tmp_path), [SYM])
    t1, t2 = idx.trades[SYM]
    assert (t1.taker_side, t1.maker_book_side, t1.price, t1.qty, t1.trade_id, t1.ts_ns) == (
        SELL, "bids", D("100.0"), D("0.4"), 1000, at(10))
    assert (t2.taker_side, t2.maker_book_side) == (BUY, "asks")
    assert isinstance(t1.price, Decimal) and t1.ord_type == "limit"
    assert idx.healthy(SYM, at(10)) and idx.segments == [(at(0), at(3_000))]
    assert not idx.healthy(SYM, at(3_001))  # after the public stream's last record: unknowable


def test_trade_index_feed_health(tmp_path):
    pub = Public()
    pub.trade(at(100), "sell", "100.0", "1")  # id 1000
    pub.trade(at(200), "sell", "100.0", "1", trade_id=1002)  # 1001 missing: (100, 200) unhealthy
    pub.trade(at(200), "sell", "100.0", "1", trade_id=1002)  # exact duplicate (resent print): dropped
    pub.trade(at(400), "sell", "100.0", "2", trade_id=1003)
    pub.trade(at(400), "sell", "100.0", "9", trade_id=1003)  # conflicting duplicate: unhealthy at 400
    pub.trade(at(500), "sideways", "100.0", "1", trade_id=1004)  # invalid aggressor side
    pub.trade(at(600), "sell", "100.0", "1", trade_id=None)  # no trade_id: continuity unknown
    pub.trade(at(700), "sell", "100.0", "1", trade_id=1005)
    idx = TradeIndex.from_files(pub.write(tmp_path), [SYM])
    assert idx.diag["duplicate_trade_id"] == 1 and idx.diag["conflicting_trade_id"] == 1
    assert idx.diag["invalid_side"] == 1 and idx.diag["missing_trade_id"] == 1 and idx.diag["trade_id_gaps"] == 1
    assert [t.trade_id for t in idx.trades[SYM]] == [1000, 1002, 1003, 1004, None, 1005]
    assert [t.taker_side for t in idx.trades[SYM]][3] is None  # never guessed
    assert idx.healthy(SYM, at(50)) and idx.healthy(SYM, at(300)) and idx.healthy(SYM, at(650))
    for ms in (150, 400, 500, 600):
        assert not idx.healthy(SYM, at(ms)), ms


def test_receive_gap_breaks_coverage(tmp_path):
    pub = Public(cover=(0,))
    pub.book(1_000)
    pub.gap()
    pub.book(2_000)
    pub.book(3_000)
    idx = TradeIndex.from_files(pub.write(tmp_path), [SYM])
    assert idx.segments == [(at(0), at(1_000)), (at(2_000), at(3_000))]
    assert idx.healthy(SYM, at(500)) and not idx.healthy(SYM, at(1_500)) and idx.healthy(SYM, at(2_500))


def test_tolerance_is_validated():
    with pytest.raises(ValueError):
        MatchTolerance(before_ns=-1)
    with pytest.raises(ValueError):
        MatchTolerance(after_ns=1.5)


# --- attribution --------------------------------------------------------------------------------------------


def test_direct_match_and_partial_coverage(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "0.6", "modify")])  # 0.4 traded
    tape.update(20, bids=[order("B2", "100.0", "2.0", "delete")])  # 0.5 traded, 1.5 canceled
    tape.update(30)
    pub = Public()
    t1 = pub.trade(at(10), "sell", "100.0", "0.4")
    t2 = pub.trade(at(20), "sell", "100.0", "0.5")
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(CONF, D("0.4"), (t1["trade_id"],))]]
    assert labels(att, "B2") == [[(CONF, D("0.5"), (t2["trade_id"],)), (ABSENT, D("1.5"), ())]]
    assert att.stats["trades"] == {f"{SYM}|MATCHED": 2}


def test_multiple_trades_against_a_level_fifo(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "0.5", "modify")])
    tape.update(30)
    pub = Public()
    a = pub.trade(at(10), "sell", "100.0", "1.0")
    b = pub.trade(at(10), "sell", "100.0", "1.5")
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(CONF, D("1.0"), (a["trade_id"],))]]
    assert labels(att, "B2") == [[(CONF, D("1.5"), (b["trade_id"],))]]


def test_one_trade_spans_several_reductions_and_is_used_once(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(10, bids=[order("B9", "100.0", "1.0", "add")])  # same µs, later item
    tape.update(10, bids=[order("B9", "100.0", "1.0", "delete")])  # competes for the same print
    tape.update(30)
    pub = Public()
    t = pub.trade(at(10), "sell", "100.0", "2.5")
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(CONF, D("1.0"), (t["trade_id"],))]]
    assert labels(att, "B2") == [[(CONF, D("1.5"), (t["trade_id"],)), (ABSENT, D("0.5"), ())]]
    assert labels(att, "B9") == [[(ABSENT, D("1.0"), ())]]  # the print was consumed in stream order
    assert sum(att.trade_used.values()) == D("2.5")


def test_cancel_without_trade_and_unhealthy_fallback(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])  # healthy feed, no print: a cancel
    tape.update(4_000, bids=[order("B2", "100.0", "2.0", "delete")])  # after the public stream ends
    tape.update(4_100, bids=[order("B3", "100.0", "1.0", "add")])
    tape.update(4_200, bids=[order("B3", "100.0", "0.5", "modify")])
    tape.update(4_300)
    _, att, _ = run(tmp_path, tape, Public(cover=(0, 3_000)))
    assert labels(att, "B1") == [[(ABSENT, D("1.0"), ())]]
    assert labels(att, "B2") == [[(DEL, D("2.0"), ())]]  # no coverage: the L3 label, never "absent"
    assert labels(att, "B3") == [[(MOD, D("0.5"), ())]]


def test_one_microsecond_late_print_is_plausible_and_configurable(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])
    tape.update(30)
    pub = Public()
    t = pub.trade(at(10) + US, "sell", "100.0", "1.0")
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(PLAUS, D("1.0"), (t["trade_id"],))]]
    _, strict, _ = run(tmp_path / "strict", tape, pub, tol=MatchTolerance(after_ns=0))
    # with no window the late print cannot explain the delete; it is a valid print, so the feed stays
    # healthy and the delete is labeled a cancel, while the print is reported unmatched
    assert labels(strict, "B1") == [[(ABSENT, D("1.0"), ())]]
    assert strict.stats["trades"] == {f"{SYM}|UNMATCHED": 1}


def test_wrong_side_price_and_ambiguous_keys_do_not_match(tmp_path):
    tape = basic_tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")])
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")], asks=[order("A1", "100.2", "1.0", "delete")])
    tape.update(30)
    pub = Public()
    pub.trade(at(10), "buy", "100.0", "1.0")  # a taker BUY hits asks, not the bid at 100.0
    pub.trade(at(10), "sell", "99.9", "1.0")  # right side, but 99.9 had no reduction
    good = pub.trade(at(10), "buy", "100.2", "1.0")  # the ask reduction at 100.2
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(ABSENT, D("1.0"), ())]]
    # the delete removes A1's tracked 3.0 (not the event's 1.0): 1.0 traded, 2.0 canceled
    assert labels(att, "A1") == [[(CONF, D("1.0"), (good["trade_id"],)), (ABSENT, D("2.0"), ())]]
    assert att.stats["trades"] == {f"{SYM}|MATCHED": 1, f"{SYM}|UNMATCHED": 2}


def test_invalid_side_print_blocks_absence_claims(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])
    tape.update(30)
    pub = Public()
    pub.trade(at(10), None, "100.0", "1.0")  # no aggressor side: never matched, never guessed
    _, att, _ = run(tmp_path, tape, pub)
    assert labels(att, "B1") == [[(DEL, D("1.0"), ())]]  # could have been this print: no "absent" claim
    assert att.stats["trades"] == {f"{SYM}|INVALID_SIDE": 1}


def test_out_of_order_and_outside_coverage_prints(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete")])
    tape.update(20, bids=[order("B2", "100.0", "2.0", "delete")])
    tape.update(30)
    pub = Public()
    b = pub.trade(at(20), "sell", "100.0", "2.0", trade_id=1001)  # recorded before the earlier print
    a = pub.trade(at(10), "sell", "100.0", "1.0", trade_id=1000)
    pub.trade(at(0) - 5 * US, "sell", "100.0", "1.0", trade_id=999)  # before the first L3 item
    _, att, idx = run(tmp_path, tape, pub)
    assert [t.trade_id for t in idx.trades[SYM]] == [999, 1000, 1001]
    assert labels(att, "B1") == [[(CONF, D("1.0"), (a["trade_id"],))]]
    assert labels(att, "B2") == [[(CONF, D("2.0"), (b["trade_id"],))]]
    assert att.stats["trades"] == {f"{SYM}|MATCHED": 2, f"{SYM}|OUTSIDE_L3": 1}


def test_only_plausible_executions_are_labeled(tmp_path):
    tape = basic_tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")])
    tape.update(10, bids=[order("C1", "99.9", "1.0", "delete")])  # not at the touch: cannot be a trade
    tape.update(20, bids=[order("B1", "100.0", "1.5", "modify")])  # an increase
    tape.update(30)
    pub = Public()
    pub.trade(at(10), "sell", "99.9", "1.0")
    _, att, _ = run(tmp_path, tape, pub)
    assert att.by_ref == {} and att.stats["trades"] == {f"{SYM}|UNMATCHED": 1}


# --- settlement ----------------------------------------------------------------------------------------------


def hand_order(qty, *candidates):
    r = PlacementRequest("r", SYM, BUY, 0)
    o = SimulatedOrder.new(r, "join_best", OrderIntent(BUY, D("100.0")), D(qty), 0)
    o.start(0, [], 1, D(1), 10_000)
    for i, (ts, l3, parts) in enumerate(candidates):
        total = sum((D(q) for q, _ in parts), D(0))
        o.add_fill(FillCandidate(ts, D("100.0"), total, Mechanism.LEVEL, l3, (i, 0), (),
                                 tuple(EvidencePart(D(q), ev) for q, ev in parts)))
    o.finish(EndReason.HORIZON, 10_000)
    return o


def test_settlement_counts_only_qualifying_slices():
    o = hand_order("1", (1, DEL, [("0.3", CONF), ("0.2", PLAUS), ("2.0", ABSENT)]), (2, MOD, [("0.4", CONF)]))
    cons = settle(o, "conservative", with_trades=True)
    assert (cons.filled_qty, [f.qty for f in cons.fills]) == (D("0.7"), [D("0.3"), D("0.4")])
    bal = settle(o, "balanced", with_trades=True)
    assert (bal.filled_qty, bal.status) == (D("0.9"), FillStatus.PARTIAL)
    assert [[(p.evidence, p.qty) for p in f.by_evidence] for f in bal.fills] == [
        [(CONF, D("0.3")), (PLAUS, D("0.2"))], [(CONF, D("0.4"))]]
    opt = settle(o, "optimistic", with_trades=True)  # TRADE_ABSENT is a cancel: never counted
    assert opt.filled_qty == D("0.9")
    assert settle(o, "optimistic").filled_qty == D("1")  # the L3-only bracket still counts the delete


def test_settlement_clips_and_uses_strongest_evidence_first():
    o = hand_order("0.5", (1, DEL, [("0.4", PLAUS), ("0.4", CONF)]))
    bal = settle(o, "balanced", with_trades=True)
    assert bal.status is FillStatus.FILLED and bal.complete_ts == 1
    assert [(p.evidence, p.qty) for p in bal.fills[0].by_evidence] == [(CONF, D("0.4")), (PLAUS, D("0.1"))]


def test_trade_settlement_needs_attribution():
    r = PlacementRequest("r", SYM, BUY, 0)
    o = SimulatedOrder.new(r, "join_best", OrderIntent(BUY, D("100.0")), D("1"), 0)
    o.start(0, [], 1, D(1), 10_000)
    o.add_fill(FillCandidate(1, D("100.0"), D("1"), Mechanism.LEVEL, MOD, (0, 0)))
    assert settle(o, "balanced").filled_qty == D("1")
    with pytest.raises(ValueError, match="attribution"):
        settle(o, "balanced", with_trades=True)


# --- replay integration ------------------------------------------------------------------------------------------


def queue_tape():
    """join_best buy 0.5 at 100.0 decided at 5 ms behind B1 1.0 and B2 2.0; B3 joins behind us."""
    tape = basic_tape()
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete")])  # ahead executes
    tape.update(30, bids=[order("B2", "100.0", "2.0", "delete")])  # ahead executes: we are first
    tape.update(40, bids=[order("B3", "100.0", "3.8", "modify")])  # 0.2 traded: it reaches us
    tape.update(50, bids=[order("B3", "100.0", "3.8", "delete")])  # 0.3 traded, the rest canceled
    tape.update(60)
    tape.update(2_000)  # data past the 1 s horizon, so the request passes the coverage filter
    return tape


def test_trades_ahead_of_us_only_advance_the_queue(tmp_path):
    tape = basic_tape()
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])
    tape.update(20, bids=[order("B1", "100.0", "0.4", "modify")])  # 0.6 traded, but B1 is ahead of us
    tape.update(30)
    pub = Public()
    pub.trade(at(20), "sell", "100.0", "0.6")
    result, att, _ = run(tmp_path, tape, pub, schedule=[req(5)])
    (o,) = result.orders
    assert labels(att, "B1") == [[(CONF, D("0.6"), (1000,))]]  # a real execution...
    assert o.fills == [] and o.tracker.ahead_qty == D("2.4")  # ...against the queue ahead: no fill
    for p in ("conservative", "balanced", "optimistic"):
        assert settle(o, p, with_trades=True).status is FillStatus.UNFILLED


def test_trades_that_reach_us_fill_and_are_clipped(tmp_path):
    pub = Public()
    for ms, qty in ((20, "1.0"), (30, "2.0"), (40, "0.2"), (50, "0.3"), (50, "0.4")):
        pub.trade(at(ms), "sell", "100.0", qty)
    result, att, _ = run(tmp_path, queue_tape(), pub, schedule=[req(5)])
    (o,) = result.orders
    # B3's delete removes 3.8: the 0.3 and 0.4 prints both cover it (0.7 confirmed), 3.1 canceled
    assert labels(att, "B3") == [[(CONF, D("0.2"), (1002,))], [(CONF, D("0.7"), (1003, 1004)), (ABSENT, D("3.1"), ())]]
    assert [(c.mechanism, c.evidence, [(p.evidence, p.qty) for p in c.attribution]) for c in o.fills] == [
        (Mechanism.LEVEL, MOD, [(CONF, D("0.2"))]),
        (Mechanism.LEVEL, DEL, [(CONF, D("0.7")), (ABSENT, D("3.1"))])]
    cons = settle(o, "conservative", with_trades=True)
    assert (cons.status, cons.first_fill_ts, cons.complete_ts) == (FillStatus.FILLED, at(40), at(50))
    assert [f.qty for f in cons.fills] == [D("0.2"), D("0.3")]  # clipped to the remaining 0.3
    assert settle(o, "conservative").status is FillStatus.UNFILLED  # the L3-only lower bound is unchanged


def test_cancel_only_never_fills_under_trade_policies(tmp_path):
    result, att, _ = run(tmp_path, queue_tape(), Public(), schedule=[req(5)])  # no prints at all
    (o,) = result.orders
    assert [[p.evidence for p in c.attribution] for c in o.fills] == [[ABSENT], [ABSENT]]
    for p in ("conservative", "balanced", "optimistic"):
        assert settle(o, p, with_trades=True).filled_qty == 0
    assert settle(o, "balanced").filled_qty == D("0.2") and settle(o, "optimistic").filled_qty == D("0.5")


def test_cross_arrival_keeps_its_l3_label(tmp_path):
    tape = basic_tape()
    tape.update(10, asks=[order("A2", "100.1", "0.2", "add")])  # crosses improve_quote's 100.1 buy
    tape.update(30)
    result, _, _ = run(tmp_path, tape, Public(), schedule=[req(5)], strategies=[ImproveQuoteStrategy()])
    (o,) = result.orders
    assert [[(p.evidence, p.qty) for p in c.attribution] for c in o.fills] == [[(CROSS, D("0.2"))]]
    assert settle(o, "optimistic", with_trades=True).filled_qty == D("0.2")
    assert settle(o, "balanced", with_trades=True).filled_qty == 0


def test_lifecycle_and_l3_labels_are_unchanged_by_attribution(tmp_path):
    tape = queue_tape()
    tape.update(70, asks=[order("A2", "100.1", "0.2", "add")])
    tape.update(80)
    pub = Public()
    for ms, qty in ((20, "1.0"), (30, "2.0"), (40, "0.2")):
        pub.trade(at(ms), "sell", "100.0", qty)
    strategies = [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()]
    schedule = [req(5), req(5, SELL), req(35), req(65)]

    def summary(result):
        return [(o.order_id, o.price, o.inserted_ts, o.end_reason, o.end_ts, o.skip_reason, o.queue_ahead_qty0,
                 [(c.ts_ns, c.qty, c.mechanism, c.evidence, c.event_ref) for c in o.fills]) for o in result.orders]

    plain, _, _ = run(tmp_path / "a", tape, pub, schedule=schedule, strategies=strategies, attribution=False)
    with_t, att, _ = run(tmp_path / "b", tape, pub, schedule=schedule, strategies=strategies)
    again, att2, _ = run(tmp_path / "c", tape, pub, schedule=schedule, strategies=strategies)
    assert summary(plain) == summary(with_t) == summary(again)  # nothing but the trade slices is added
    assert att.by_ref == att2.by_ref and att.stats == att2.stats  # deterministic
    assert [[c.attribution for c in o.fills] for o in with_t.orders] == [[c.attribution for c in o.fills]
                                                                       for o in again.orders]
    assert all(c.attribution is None for o in plain.orders for c in o.fills)
    assert any(c.attribution for o in with_t.orders for c in o.fills)


def test_misaligned_attribution_is_rejected(tmp_path):
    other = basic_tape()
    other.update(10)
    other.update(60)
    l3_other = other.write(tmp_path / "other")
    idx = TradeIndex.from_files(Public().write(tmp_path / "pub"), [SYM])
    att = attribute_trades(engine([]).cfg, idx, files=l3_other)  # labels for a different recording
    eng = engine([req(5)])
    eng.on_reach = FillRecorder(att)
    with pytest.raises(ValueError, match="same L3 files"):
        eng.run(queue_tape().write(tmp_path / "l3"))


def test_plausible_execution_matches_the_tracker_reach_rule(tmp_path):
    """Every LEVEL / INTERCEPT reach is a labeled plausible execution (FillRecorder would raise otherwise)."""
    result, att, _ = run(tmp_path, queue_tape(), Public(), schedule=[req(5), req(5, SELL)],
                         strategies=[JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()])
    refs = {c.event_ref for o in result.orders for c in o.fills if c.mechanism is not Mechanism.CROSS_ARRIVAL}
    assert refs and refs <= set(att.by_ref)
    events = [r for r in att.by_ref.values()]
    assert all(r.volume > 0 and r.kind in ("modify", "delete") for r in events)


def test_is_plausible_execution_rule():
    from toxicflow.execsim.sim_book import SimL3Book
    from execsim_helpers import BTC, Sealer
    sealer, book = Sealer(), SimL3Book(BTC, 10)
    book.apply("snapshot", sealer.snapshot(bids=BIDS + [order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "1")]))
    book.apply("update", sealer.update(bids=[order("B1", "100.0", "0.5", "modify"), order("C1", "99.9", "1.0", "delete"),
                                             order("B2", "100.0", "3.0", "modify"), order("B1", "100.1", "0.5", "modify")]))
    assert [is_plausible_execution(ev) for ev in book.last_events] == [True, False, False, False]


# --- analytics -----------------------------------------------------------------------------------------------


def test_metrics_in_both_evidence_modes(tmp_path):
    pub = Public()
    for ms, qty in ((20, "1.0"), (30, "2.0"), (40, "0.2")):
        pub.trade(at(ms), "sell", "100.0", qty)
    result, att, _ = run(tmp_path, queue_tape(), pub, schedule=[req(5)])
    cfg = engine([]).cfg
    l3 = analyze(result, cfg, strategies=["join_best"])
    tr = analyze(result, cfg, strategies=["join_best"], evidence="trades", attribution=att)
    assert l3.evidence_mode == "l3_only" and tr.evidence_mode == "trades"
    assert l3.lifecycle == tr.lifecycle and l3.queue == tr.queue  # placement and queue never depend on evidence
    s_l3, s_tr = l3.summary[("join_best", "optimistic")], tr.summary[("join_best", "optimistic")]
    assert (s_l3.filled_full, s_l3.fill_qty_by_evidence) == (1, {"L3_DELETE_AMBIGUOUS": D("0.3"),
                                                                  "L3_MODIFY_DOWN": D("0.2")})
    assert (s_tr.filled_any, s_tr.filled_full, s_tr.fill_qty_by_evidence) == (1, 0, {"TRADE_CONFIRMED": D("0.2")})
    assert s_tr.fills_by_evidence == {"TRADE_CONFIRMED": 1}
    assert tr.summary[("join_best", "conservative")].filled_any == 1 > l3.summary[("join_best", "conservative")].filled_any
    ev = tr.evidence["join_best"]
    assert (ev.n_candidates, ev.candidates_by_evidence, ev.qty_by_evidence) == (
        2, {"TRADE_ABSENT": 1, "TRADE_CONFIRMED": 1}, {"TRADE_ABSENT": D("3.8"), "TRADE_CONFIRMED": D("0.2")})
    assert l3.evidence["join_best"].qty_by_evidence == {"L3_DELETE_AMBIGUOUS": D("3.8"), "L3_MODIFY_DOWN": D("0.2")}
    assert tr.attribution == att.stats and l3.attribution is None
    assert "evidence trades" in format_report(tr) and "trade attribution" in format_report(tr)
    # Kaplan-Meier and markouts still run in trade mode
    assert tr.summary[("join_best", "conservative")].km_first_fill.n_events == 1


def test_no_trade_feed(tmp_path):
    result, _, _ = run(tmp_path, queue_tape(), None, schedule=[req(5)], attribution=False)
    cfg = engine([]).cfg
    rep = analyze(result, cfg)  # the Stage 4 path is unchanged
    assert rep.evidence_mode == "l3_only" and rep.attribution is None
    assert rep.summary[("join_best", "optimistic")].filled_full == 1
    with pytest.raises(ValueError, match="attribution"):
        analyze(result, cfg, evidence="trades")
    with pytest.raises(ValueError):
        analyze(result, cfg, evidence="maybe")
