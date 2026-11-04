"""SimulatedOrder life cycle and its tie-in with QueueTracker."""

from decimal import Decimal

import pytest
from execsim_helpers import order
from test_execsim_queue_tracker import standard

from toxicflow.execsim.models import (
    EndReason,
    Evidence,
    FillCandidate,
    Mechanism,
    OrderIntent,
    PlacementRequest,
    Side,
    SkipReason,
)
from toxicflow.execsim.orders import OrderState, SimulatedOrder

D = Decimal
REQ = PlacementRequest("r1", "BTC/USD", Side.BUY, 1_000)
H = 10_000


def new(price="100.0", qty="0.5"):
    return SimulatedOrder.new(REQ, "join_best", OrderIntent(Side.BUY, D(price)), D(qty), decided_ts=REQ.ts_ns)


def candidate(qty, evidence, price="100.0"):
    return FillCandidate(1_500, D(price), D(qty), Mechanism.LEVEL, evidence, (1, 0))


def test_new_order_is_pending():
    o = new()
    assert (o.order_id, o.state, o.side, o.book_side, o.price, o.qty) == (
        "r1/join_best", OrderState.PENDING, Side.BUY, "bids", D("100.0"), D("0.5"))
    assert o.tracker is None and o.fills == [] and not o.censored


@pytest.mark.parametrize(("intent", "qty"), [(OrderIntent(Side.SELL, D("100.0")), D("1")),
                                            (OrderIntent(Side.BUY, D("100.0")), D("0"))])
def test_new_rejects_bad_input(intent, qty):
    with pytest.raises(ValueError):
        SimulatedOrder.new(REQ, "join_best", intent, qty, decided_ts=0)


def test_start_records_queue_and_horizon():
    m = standard()
    o = new()
    o.start(inserted_ts=1_000, ahead=m.book.level_orders("bids", D("100.0")), rank=1, touch_qty=D("3.0"),
            horizon_ns=H)
    assert o.state is OrderState.LIVE and (o.inserted_ts, o.expiry_ts) == (1_000, 11_000)
    assert (o.queue_ahead_qty0, o.queue_ahead_orders0, o.touch_qty0, o.level_rank0) == (D("3.0"), 2, D("3.0"), 1)
    assert list(o.tracker.ahead_ids) == ["B1", "B2"] and o.order_id not in m.book.orders
    with pytest.raises(RuntimeError):
        o.start(inserted_ts=1_000, ahead=(), rank=1, touch_qty=D(0), horizon_ns=H)
    with pytest.raises(ValueError):
        new().start(inserted_ts=1_000, ahead=(), rank=1, touch_qty=D(0), horizon_ns=0)


def test_skip_only_from_pending():
    o = new()
    with pytest.raises(RuntimeError):
        o.finish(EndReason.HORIZON, 2_000)
    o.skip(SkipReason.CROSSED_AT_INSERT)
    assert o.state is OrderState.ENDED and o.skip_reason is SkipReason.CROSSED_AT_INSERT and o.end_reason is None
    with pytest.raises(RuntimeError):
        o.skip(SkipReason.DATA_GAP)


def test_fills_are_clipped_per_evidence_threshold():
    o = new()
    o.start(inserted_ts=1_000, ahead=(), rank=1, touch_qty=D(0), horizon_ns=H)
    o.add_fill(candidate("0.2", Evidence.L3_MODIFY_DOWN))
    o.add_fill(candidate("4.0", Evidence.L3_DELETE_AMBIGUOUS))
    assert o.filled_qty(None) == 0 and o.remaining_qty(None) == D("0.5")  # conservative, L3 only
    assert o.filled_qty(Evidence.L3_MODIFY_DOWN) == D("0.2") and o.remaining_qty(3) == D("0.3")  # balanced
    assert o.filled_qty(Evidence.L3_CROSS_ARRIVAL) == D("0.5")  # optimistic: clipped to qty
    assert [f.qty for f in o.fills] == [D("0.2"), D("4.0")]  # logged unclipped
    with pytest.raises(ValueError):
        o.add_fill(candidate("0.1", Evidence.L3_MODIFY_DOWN, price="99.9"))


@pytest.mark.parametrize(("reason", "censored"), [(EndReason.HORIZON, False), (EndReason.ALL_POLICIES_FILLED, False),
                                                  (EndReason.CENSORED_RESNAPSHOT, True),
                                                  (EndReason.OUT_OF_DEPTH, True)])
def test_finish_closes_tracker_and_keeps_fills(reason, censored):
    o = new()
    o.start(inserted_ts=1_000, ahead=(), rank=1, touch_qty=D(0), horizon_ns=H)
    o.add_fill(candidate("0.2", Evidence.L3_MODIFY_DOWN))
    o.finish(reason, end_ts=5_000)
    assert o.state is OrderState.ENDED and (o.end_reason, o.end_ts, o.censored) == (reason, 5_000, censored)
    assert o.tracker.invalidated is reason and o.filled_qty(3) == D("0.2")
    with pytest.raises(RuntimeError):
        o.add_fill(candidate("0.1", Evidence.L3_MODIFY_DOWN))
    with pytest.raises(RuntimeError):
        o.finish(EndReason.HORIZON, 6_000)


def test_order_through_queue_to_candidate_fills():  # scenario 5, evidence assigned by hand
    m = standard()
    o = new()
    o.start(inserted_ts=0, ahead=m.book.level_orders("bids", D("100.0")), rank=1, touch_qty=D("3.0"), horizon_ns=H)
    m.feed(o.tracker, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    m.feed(o.tracker, bids=[order("B3", "100.0", "4.0", "add")])
    for qty in ("3.8", "3.2"):
        for r in m.feed(o.tracker, bids=[order("B3", "100.0", qty, "modify")]):
            # stand-in for ExecutionClassifier + FillSimulator.record (not built yet)
            o.add_fill(FillCandidate(r.item_ts_ns, o.price, r.volume, r.mechanism, Evidence.L3_MODIFY_DOWN, r.ref))
    assert [f.qty for f in o.fills] == [D("0.2"), D("0.6")]
    assert o.filled_qty(None) == 0 and o.filled_qty(3) == D("0.5") and o.filled_qty(1) == D("0.5")
    o.finish(EndReason.HORIZON, end_ts=H)
    assert o.diag["fifo_anomaly"] == 0
