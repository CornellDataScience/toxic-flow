"""Seeded property tests for invariants I1 to I7 (design §6.1, §10.4) on random synthetic streams.

Each seed builds a random L3 recording (adds, partial and full reductions, qty
increases, price moves, re-adds, crossing arrivals, equal and regressing
timestamps, checksum failures, resnapshots, file boundaries, receive gaps, depth
truncation) and replays it through the engine with random requests, horizons,
latencies and increase policies. A plain L3Book replays the same records as the
reference.

I5 and I6 are policy properties of FillSimulator, which does not exist yet. Here
they are checked on SimulatedOrder.filled_qty with a stand-in L3-only labeling
(modify -> L3_MODIFY_DOWN, delete -> L3_DELETE_AMBIGUOUS, cross -> L3_CROSS_ARRIVAL)
and the §8.7 rank thresholds, so the candidate stream the engine produces is
shown to satisfy them under clipping.
"""

import itertools
import random
from decimal import Decimal

import pytest
from execsim_helpers import BTC, order
from test_execsim_replay import SYM, Tape, at

from toxicflow.execsim.config import SimConfig
from toxicflow.execsim.models import Evidence, FillCandidate, Mechanism, OrderIntent, PlacementRequest, Side, Skip, \
    SkipReason
from toxicflow.execsim.orders import OrderState, SimulatedOrder
from toxicflow.execsim.queue_tracker import QueueTracker
from toxicflow.execsim.reader import KIND_META, RecordingReader, book_items
from toxicflow.execsim.replay import HistoricalReplayEngine
from toxicflow.execsim.results import DISCONTINUITY_KINDS
from toxicflow.execsim.strategies import JoinBestStrategy
from toxicflow.execsim.timeutil import MS, parse_rfc3339_ns
from toxicflow.kraken.book import L3Book

D = Decimal
QTYS = [D("0.1"), D("0.3"), D("0.5"), D("1"), D("2")]
# (conservative, balanced, optimistic) thresholds from §8.7, L3-only and with trades
THRESHOLDS = {"l3_only": (None, Evidence.L3_MODIFY_DOWN, Evidence.L3_CROSS_ARRIVAL),
              "with_trades": (Evidence.TRADE_CONFIRMED, Evidence.TRADE_PLAUSIBLE, Evidence.L3_CROSS_ARRIVAL)}


class SecondLevel:
    """Test-only strategy: join the second-best level, to exercise deeper queues and intercepts."""

    name = "second_level"

    def decide(self, req, view):
        levels = view.bids if req.side is Side.BUY else view.asks
        return OrderIntent(req.side, levels[1].price) if len(levels) > 1 else Skip(SkipReason.NO_TOUCH)


def book_orders(book, name):
    return [order(oid, str(price), str(qty)) for price, level in book.side(name).items() for oid, qty in level.items()]


def random_events(rng, book, name, new_id):
    """1 to 3 events on one side, chosen from the reference state before the item."""
    base = D("99.6") if name == "bids" else D("100.1")
    prices = [base + D("0.1") * i for i in range(6)]
    known = [oid for oid, (side, _) in book.orders.items() if side == name]
    out = []
    for _ in range(rng.randint(1, 3)):
        r = rng.random()
        if not known or r < 0.35:
            price = rng.choice(prices)
            if name == "asks" and rng.random() < 0.08 and book.bids:  # an arrival through the bids
                price = book.bids.peekitem(0)[0] - D("0.1") * rng.randint(0, 2)
            out.append(order(new_id(name[0].upper()), str(price), str(rng.choice(QTYS)), "add"))
            continue
        oid = rng.choice(known)
        _, price = book.orders[oid]
        qty = book.side(name)[price][oid] if price in book.side(name) and oid in book.side(name)[price] else D("1")
        if r < 0.55:  # partial reduction
            new = (qty * D(rng.choice(["0.2", "0.5", "0.8"]))).quantize(D("0.01"))
            out.append(order(oid, str(price), str(new), "modify") if new > 0 else order(oid, str(price), "0", "delete"))
        elif r < 0.75:
            out.append(order(oid, str(price), str(qty), "delete"))
        elif r < 0.83:  # qty increase
            out.append(order(oid, str(price), str(qty + rng.choice(QTYS)), "modify"))
        elif r < 0.91:  # price move: loses priority
            out.append(order(oid, str(rng.choice(prices)), str(qty), "modify"))
        else:  # re-add of a known ID: back of the queue
            out.append(order(oid, str(rng.choice([price, rng.choice(prices)])), str(qty), "add"))
    return out


def random_tape(seed, steps=300):
    rng = random.Random(seed)
    tape = Tape(depth=rng.choice([4, 10]))
    ids = itertools.count()

    def new_id(prefix):
        return f"{prefix}{next(ids)}"

    bids = [order(new_id("B"), str(D("99.7") + D("0.1") * i), str(rng.choice(QTYS))) for i in range(4) for _ in "ab"]
    asks = [order(new_id("A"), str(D("100.2") + D("0.1") * i), str(rng.choice(QTYS))) for i in range(4) for _ in "ab"]
    tape.snapshot(0, bids=bids, asks=asks, recv_ms=0)
    ms = recv_shift = 0
    for _ in range(steps):
        ms += rng.choice([0, 0, 1, 1, 2, 3, 5, 8])  # many equal timestamps
        if rng.random() < 0.01:
            ms = max(0, ms - 3)  # a timestamp regression
        r = rng.random()
        book = tape.sealer.book
        if r < 0.025:  # resync while (usually) synced
            tape.snapshot(ms, bids=book_orders(book, "bids"), asks=book_orders(book, "asks"), recv_ms=ms + recv_shift)
            continue
        if r < 0.035:  # hourly rotation: the new file starts with updates, then a snapshot
            tape.new_file()
        elif r < 0.045:
            recv_shift += 6_000  # receive gap before the next record
        sides = {name: random_events(rng, book, name, new_id) for name in ("bids", "asks") if rng.random() < 0.6}
        corrupt = rng.random() < 0.03
        tape.update(ms, bids=sides.get("bids", ()), asks=sides.get("asks", ()), corrupt=corrupt,
                    recv_ms=ms + recv_shift)
        if r < 0.035 or corrupt and rng.random() < 0.7:
            book = tape.sealer.book
            ms += rng.choice([0, 1, 4])
            tape.snapshot(ms, bids=book_orders(book, "bids"), asks=book_orders(book, "asks"), recv_ms=ms + recv_shift)
    return tape, ms, rng


def label(mechanism, kind):
    if mechanism is Mechanism.CROSS_ARRIVAL:
        return Evidence.L3_CROSS_ARRIVAL
    return Evidence.L3_MODIFY_DOWN if kind == "modify" else Evidence.L3_DELETE_AMBIGUOUS


def random_setup(seed, tmp_path):
    """A random recording on disk plus a random schedule and configuration for it."""
    tape, end_ms, rng = random_tape(seed)
    files = tape.write(tmp_path)
    schedule = [PlacementRequest(f"r{i}", SYM, rng.choice([Side.BUY, Side.SELL]), at(rng.uniform(-5, end_ms + 20)))
                for i in range(60)]
    cfg = SimConfig(symbols=(SYM,), order_qty={SYM: rng.choice(["0.1", "1", "5"])},
                    horizon_ns=rng.choice([3, 10, 40, 200]) * MS, latency_ns=rng.choice([0, 0, 2 * MS]),
                    warmup_ns=rng.choice([0, 0, 5 * MS]), increase_policy=rng.choice(["keep", "back"]),
                    audit_invariants=True)
    return files, cfg, schedule


def replay_with_reference(seed, tmp_path, monkeypatch):
    files, cfg, schedule = random_setup(seed, tmp_path)

    obs = {"starts": [], "seen": [], "reaches": [], "cumulative": {}}
    state = {"ref": None, "seq": None}

    real_start = SimulatedOrder.start

    def start(self, inserted_ts, ahead, rank, touch_qty, **kw):
        ref = state["ref"]  # the reference book before the record being processed
        obs["starts"].append((self.order_id, state["seq"], tuple(ahead), tuple(ref.side(self.book_side)
                              .get(self.price, {}).items()), rank, ref.side(self.book_side).bisect_left(self.price) + 1,
                              touch_qty))
        real_start(self, inserted_ts, ahead, rank, touch_qty, **kw)

    real_on_event = QueueTracker.on_event

    def on_event(self, ev, item_ts_ns=None):
        obs["seen"].append(ev.position.seq)
        return real_on_event(self, ev, item_ts_ns)

    monkeypatch.setattr(SimulatedOrder, "start", start)
    monkeypatch.setattr(QueueTracker, "on_event", on_event)

    def on_reach(o, reach, ev):
        obs["reaches"].append((o, reach, ev))
        o.add_fill(FillCandidate(reach.item_ts_ns, o.price, reach.volume, reach.mechanism,
                                 label(reach.mechanism, ev.kind), reach.ref))
        now = {name: tuple(o.filled_qty(t) for t in ts) for name, ts in THRESHOLDS.items()}
        before = obs["cumulative"].get(o.order_id)
        for name, (c, b, opt) in now.items():
            assert D(0) <= c <= b <= opt <= o.qty  # I5 bound and I6 ordering
            if before is not None:
                assert all(x >= y for x, y in zip(now[name], before[name]))  # I5: never decreases
        obs["cumulative"][o.order_id] = now

    eng = HistoricalReplayEngine(cfg, [JoinBestStrategy(), SecondLevel()], schedule, on_reach=on_reach)
    expected, failed, parity = {"checksum_ok": 0, "checksum_fail": 0, "skipped_unsynced": 0}, set(), []
    last_ts = {"v": 0}
    for rec in RecordingReader(files=files, symbols=cfg.symbols, gap_ns=cfg.gap_ns):
        if rec.kind == KIND_META:
            state["ref"] = L3Book(BTC, rec.payload["depth"])
        state["seq"] = rec.seq
        eng.process_record(rec)
        for msg_type, item in book_items(rec):
            last_ts["v"] = max(last_ts["v"], parse_rfc3339_ns(item["timestamp"]))
            ok = state["ref"].apply(msg_type, item)
            expected[{True: "checksum_ok", False: "checksum_fail", None: "skipped_unsynced"}[ok]] += 1
            if ok is not True:
                failed.add(rec.seq)
            sim, ref = eng.state[SYM].book, state["ref"]
            parity.append((sim.bids, sim.asks, sim.orders, sim.synced) == (ref.bids, ref.asks, ref.orders,
                                                                             ref.synced))
    result = eng.finish()
    return eng, result, cfg, schedule, obs, expected, failed, parity


@pytest.mark.parametrize("seed", range(30))
def test_invariants_on_random_streams(seed, tmp_path, monkeypatch):
    eng, result, cfg, schedule, obs, expected, failed, parity = replay_with_reference(seed, tmp_path, monkeypatch)
    counters = result.counters[SYM]

    # I1: the historical book after every item equals a plain L3Book replay, with the same checksum results
    assert parity and all(parity)
    assert {k: counters[k] for k in expected} == expected

    # I2: at insertion the ahead set is exactly the level, in FIFO order; rank and touch from the same book
    for _, _, ahead, level, rank, ref_rank, touch in obs["starts"]:
        assert ahead == level and rank == ref_rank and touch >= 0

    # I3: audit mode checked every live order after every trusted item
    assert counters["audit_fail"] == 0
    assert not any(o.diag["ahead_qty_mismatch"] for o in result.orders)

    # I4: every candidate is one historical event of a trusted item, with exactly its volume
    seen_refs = set()
    insert_seq = {oid: seq for oid, seq, *_ in obs["starts"]}
    breaks = [q.seq for q in result.quality if q.kind in DISCONTINUITY_KINDS]
    for o, reach, ev in obs["reaches"]:
        assert reach.ref == (ev.position.seq, ev.event_index)
        assert (o.order_id, reach.ref) not in seen_refs  # never the same event twice for one order
        seen_refs.add((o.order_id, reach.ref))
        if reach.mechanism is Mechanism.CROSS_ARRIVAL:
            assert ev.kind == "add" and reach.volume == ev.qty == ev.qty_after and reach.event_price == ev.price
            assert ev.price <= o.price if o.side is Side.BUY else ev.price >= o.price
        else:
            assert ev.kind in ("modify", "delete") and not ev.requeued and not ev.dropped_by_truncation
            assert reach.volume == ev.qty_before - (ev.qty_after or 0) == ev.reduction > 0
            assert reach.event_price == ev.price_before
            assert (reach.event_price == o.price) is (reach.mechanism is Mechanism.LEVEL)
            if reach.mechanism is Mechanism.INTERCEPT:
                assert o.side.better(o.price, reach.event_price)
        # timing: inside the order's life, never across a discontinuity after insertion
        assert o.inserted_ts <= reach.item_ts_ns < o.expiry_ts and reach.item_ts_ns <= o.end_ts
        assert not any(insert_seq[o.order_id] <= b < ev.position.seq for b in breaks)
    assert all(r.ahead_qty == 0 for r in result.reaches)
    assert len(result.reaches) == len(obs["reaches"])

    # I7: no tracker ever saw an event from an item whose checksum failed
    assert not set(obs["seen"]) & failed

    # every request x strategy ends exactly once, as an order end or a skip
    assert len(result.orders) + len(result.skips) == len(schedule) * 2
    for o in result.orders:
        assert o.state is OrderState.ENDED and (o.end_reason is None) != (o.skip_reason is None)
        assert (o.inserted_ts is not None) is (o.end_reason is not None)
        if o.end_reason is not None:
            assert o.inserted_ts <= o.end_ts <= o.expiry_ts
            assert o.checksum_covered is (o.max_level_rank <= 10)
    # incidents: every discontinuity belongs to one, numbered in order; nothing else has one
    numbers = [q.incident for q in result.quality if q.kind in DISCONTINUITY_KINDS]
    assert None not in numbers and numbers == sorted(numbers)
    assert all(q.incident is None for q in result.quality if q.kind not in DISCONTINUITY_KINDS)


def test_random_streams_exercise_every_path(tmp_path_factory):
    """Guard against a generator that silently stops covering the interesting cases."""
    totals = {"mechanisms": set(), "ends": set(), "skips": set(), "diag": set(), "quality": set()}
    for seed in range(30):
        mp = pytest.MonkeyPatch()
        try:
            _, result, *_ = replay_with_reference(seed, tmp_path_factory.mktemp(f"s{seed}"), mp)
        finally:
            mp.undo()
        totals["mechanisms"] |= {r.reach.mechanism for r in result.reaches}
        totals["ends"] |= {o.end_reason for o in result.orders if o.end_reason}
        totals["skips"] |= {s.reason for s in result.skips} | {o.skip_reason for o in result.orders if o.skip_reason}
        totals["diag"] |= {k for o in result.orders for k, v in o.diag.items() if v}
        totals["quality"] |= {q.kind for q in result.quality}
    assert totals["mechanisms"] == set(Mechanism)
    assert {"HORIZON", "CENSORED_DESYNC", "CENSORED_RESNAPSHOT", "CENSORED_RECV_GAP",
            "CENSORED_SESSION_RESTART", "CENSORED_END_OF_DATA", "OUT_OF_DEPTH"} <= {e.value for e in totals["ends"]}
    assert {"UNSYNCED", "DATA_GAP", "NO_DATA"} <= {s.value for s in totals["skips"]}
    assert {"behind_reduction", "ahead_requeued", "ahead_increase"} <= totals["diag"]
    assert {"DESYNC", "RESNAPSHOT", "RECV_GAP", "SESSION_RESTART", "TS_REGRESSION"} <= totals["quality"]


# --- batch versus incremental -----------------------------------------------------------------


def outcome(eng, result):
    """Everything a run produces, in order, plus the final historical book."""
    orders = [
        (o.order_id, o.state, o.side, o.price, o.qty, o.decided_ts, o.inserted_ts, o.expiry_ts, o.end_ts, o.end_reason,
         o.skip_reason, o.queue_ahead_qty0, o.queue_ahead_orders0, o.touch_qty0, o.level_rank0, o.max_level_rank,
         o.checksum_covered, dict(o.diag), list(o.fills),
         None if o.tracker is None else (list(o.tracker.ahead_ids.items()), o.tracker.advanced_qty, o.tracker.front_ref,
                                         o.tracker.front_ts_ns, o.tracker.invalidated, dict(o.tracker.diag)))
        for o in result.orders
    ]
    book = eng.state[SYM].book
    final = None if book is None else (book.bids, book.asks, book.orders, book.synced, book.event_counts,
                                       book.checksum())
    return orders, result.reaches, result.skips, result.quality, result.views, result.counters, final


@pytest.mark.parametrize("seed", range(10))
def test_batch_and_incremental_replay_agree(seed, tmp_path):
    files, cfg, schedule = random_setup(seed, tmp_path)

    def fresh():
        return HistoricalReplayEngine(cfg, [JoinBestStrategy(), SecondLevel()], schedule)

    records = list(RecordingReader(files=files, symbols=cfg.symbols, gap_ns=cfg.gap_ns))
    by_run = fresh()
    outcomes = [outcome(by_run, by_run.run(files))]
    by_replay = fresh()
    outcomes.append(outcome(by_replay, by_replay.replay(iter(records))))
    by_record = fresh()
    for rec in records:
        by_record.process_record(rec)
    outcomes.append(outcome(by_record, by_record.finish()))
    # incremental with the finish deferred and called twice: finish is idempotent
    by_parts = fresh()
    cut = len(records) // 3
    for chunk in (records[:cut], records[cut:]):
        for rec in chunk:
            by_parts.process_record(rec)
    by_parts.finish()
    outcomes.append(outcome(by_parts, by_parts.finish()))

    assert outcomes[0][0] and outcomes[0][3]  # the run placed orders and logged quality events
    for other in outcomes[1:]:
        for got, want in zip(other, outcomes[0]):
            assert got == want
