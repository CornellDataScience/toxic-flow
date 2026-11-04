"""QueueTracker against BookEvents produced by a real SimL3Book (scenarios from design §6.4 and §10.2)."""

from decimal import Decimal
from pathlib import Path

import pytest
from execsim_helpers import BTC, Sealer, order

from toxicflow.execsim.models import EndReason, Mechanism, Side
from toxicflow.execsim.queue_tracker import QueueTracker
from toxicflow.execsim.reader import KIND_MSG, StreamRecord
from toxicflow.execsim.sim_book import SimL3Book

D = Decimal
BUY, SELL = Side.BUY, Side.SELL


class Market:
    """A synced SimL3Book fed by a Sealer; every update is stamped with its own seq."""

    def __init__(self, bids=(), asks=(), depth=10):
        self.sealer = Sealer(BTC, depth)
        self.book = SimL3Book(BTC, depth)
        self.seq = 0
        assert self.book.apply("snapshot", self.sealer.snapshot(bids=bids, asks=asks)) is True

    def step(self, bids=(), asks=()):
        self.seq += 1
        rec = StreamRecord(Path("synthetic"), self.seq, self.seq, 0, KIND_MSG, {})
        assert self.book.apply("update", self.sealer.update(bids=bids, asks=asks), position=rec) is True
        return self.book.last_events

    def track(self, side, price, increase_policy="keep"):
        return QueueTracker(side, D(price), self.book.level_orders(side.book_side, D(price)), increase_policy)

    def feed(self, trackers, bids=(), asks=()):
        """Apply one item; every tracker consumes its events. Returns the candidates per tracker."""
        events = self.step(bids, asks)
        if isinstance(trackers, QueueTracker):
            return trackers.on_events(events, item_ts_ns=self.seq * 1000)
        return [t.on_events(events, item_ts_ns=self.seq * 1000) for t in trackers]


def standard():
    """§10.2 book: bids 100.0: B1 1.0, B2 2.0 | asks 100.2: A1 3.0."""
    return Market(bids=[order("B1", "100.0", "1.0"), order("B2", "100.0", "2.0")], asks=[order("A1", "100.2", "3.0")])


def assert_consistent(m, t):
    assert t.audit(m.book.level_orders(t.side.book_side, t.price)) == []


# --- placement and advancement -------------------------------------------


def test_placement_behind_existing_queue_and_new_orders_go_behind():  # scenario 1
    m = standard()
    t = m.track(BUY, "100.0")
    assert list(t.ahead_ids.items()) == [("B1", D("1.0")), ("B2", D("2.0"))]
    assert (t.ahead_qty, t.ahead_orders, t.at_front) == (D("3.0"), 2, False)
    assert (t.initial_ahead_qty, t.initial_ahead_orders) == (D("3.0"), 2)

    assert m.feed(t, bids=[order("B4", "100.0", "7.0", "add")]) == []
    assert "B4" not in t.ahead_ids and t.ahead_qty == D("3.0")
    assert [oid for oid, _ in m.book.level_orders("bids", D("100.0"))] == ["B1", "B2", "B4"]
    assert_consistent(m, t)


def test_placement_on_empty_level_is_at_front():
    m = standard()
    t = m.track(BUY, "100.1")  # improve_quote: no historical orders at that price
    assert t.at_front and t.ahead_qty == 0 and t.front_ref is None


def test_partial_reduction_ahead_advances_but_is_not_a_fill():  # scenario 2
    m = standard()
    t = m.track(BUY, "100.0")
    # B1 is at the touch, so this reduction may well be a trade, but it is consumed ahead of us
    assert m.feed(t, bids=[order("B1", "100.0", "0.4", "modify")]) == []
    assert list(t.ahead_ids.items()) == [("B1", D("0.4")), ("B2", D("2.0"))]
    assert t.ahead_qty == D("2.4") and t.advanced_qty == D("0.6")
    assert_consistent(m, t)


def test_deletions_ahead_reach_front_without_fills():  # scenario 4
    m = standard()
    t = m.track(BUY, "100.0")
    assert m.feed(t, bids=[order("B1", "100.0", "1.0", "delete")]) == []
    assert t.ahead_qty == D("2.0") and not t.at_front and t.front_ref is None
    assert m.feed(t, bids=[order("B2", "100.0", "2.0", "delete")]) == []
    assert t.at_front and t.ahead_qty == 0 and t.advanced_qty == D("3.0")
    assert (t.front_ref, t.front_ts_ns) == ((2, 0), 2000)


@pytest.mark.parametrize(
    ("policy", "ahead_qty", "b1_ahead"), [("keep", D("3.5"), True), ("back", D("2.0"), False)]
)
def test_qty_increase_ahead_follows_increase_policy(policy, ahead_qty, b1_ahead):  # scenario 3
    m = standard()
    t = m.track(BUY, "100.0", increase_policy=policy)
    assert m.feed(t, bids=[order("B1", "100.0", "1.5", "modify")]) == []
    assert t.ahead_qty == ahead_qty and ("B1" in t.ahead_ids) is b1_ahead
    assert t.diag["ahead_increase"] == 1
    assert_consistent(m, t)  # I3 holds under both policies (back leaves a subset)


def test_invalid_increase_policy():
    with pytest.raises(ValueError):
        QueueTracker(BUY, D("100.0"), (), "front")


def test_worked_example_section_6_4():
    m = Market(bids=[order("B1", "100.0", "1.0"), order("B2", "100.0", "2.0")], asks=[order("A1", "100.2", "3.0")])
    t = m.track(BUY, "100.0")
    steps = [
        ([order("B3", "100.0", "4.0", "add")], D("3.0"), []),
        ([order("B1", "100.0", "0.6", "modify")], D("2.6"), []),
        ([order("B2", "100.0", "2.0", "delete")], D("0.6"), []),
        ([order("B1", "100.0", "0.6", "delete")], D("0"), []),
        ([order("B3", "100.0", "3.8", "modify")], D("0"), [D("0.2")]),
        ([order("B3", "100.0", "3.2", "modify")], D("0"), [D("0.6")]),
        ([order("B5", "100.0", "1.0", "add")], D("0"), []),
    ]
    for bids, ahead_qty, volumes in steps:
        reaches = m.feed(t, bids=bids)
        assert t.ahead_qty == ahead_qty
        assert [r.volume for r in reaches] == volumes
        assert all(r.mechanism is Mechanism.LEVEL and r.event_price == D("100.0") for r in reaches)
        assert_consistent(m, t)
    assert t.front_ref == (4, 0) and t.diag["fifo_anomaly"] == 0


# --- reach test ------------------------------------------------------------


def test_delete_at_front_is_a_level_candidate_with_full_volume():  # scenario 6
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    m.feed(t, bids=[order("B3", "100.0", "4.0", "add")])
    (r,) = m.feed(t, bids=[order("B3", "100.0", "4.0", "delete")])
    assert (r.volume, r.mechanism, r.event_price, r.ref, r.item_ts_ns) == (D("4.0"), Mechanism.LEVEL, D("100.0"),
                                                                         (3, 0), 3000)


def test_reductions_behind_us_while_queue_ahead_are_fifo_anomalies():  # scenario 7
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B4", "100.0", "7.0", "add")])
    assert m.feed(t, bids=[order("B4", "100.0", "5.0", "modify")]) == []
    assert m.feed(t, bids=[order("B4", "100.0", "5.0", "delete")]) == []
    assert t.diag["fifo_anomaly"] == 2 and t.ahead_qty == D("3.0")


def test_unrelated_levels_and_opposite_side_do_not_move_the_queue():
    m = Market(
        bids=[order("B1", "100.0", "1.0"), order("C1", "99.9", "1.0"), order("D1", "99.5", "2.0")],
        asks=[order("A1", "100.2", "3.0"), order("A2", "100.3", "1.0")],
    )
    t = m.track(BUY, "99.9")
    before = dict(t.ahead_ids)
    assert m.feed(t, bids=[order("B1", "100.0", "0.5", "modify"), order("D1", "99.5", "1.0", "modify"),
                           order("E1", "99.0", "1.0", "add"), order("B9", "100.0", "1.0", "add")]) == []
    assert m.feed(t, asks=[order("A1", "100.2", "1.0", "modify"), order("A2", "100.3", "1.0", "delete"),
                           order("A3", "100.4", "2.0", "add")]) == []
    assert t.ahead_ids == before and t.advanced_qty == 0 and sum(t.diag.values()) == 0


def test_reduction_away_from_the_touch_is_not_a_candidate():  # scenario 8
    m = Market(bids=[order("B1", "100.0", "1.0"), order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")])
    t = m.track(BUY, "99.9")
    m.feed(t, bids=[order("C1", "99.9", "1.0", "delete")])
    assert t.at_front
    m.feed(t, bids=[order("C2", "99.9", "2.0", "add")])
    assert m.feed(t, bids=[order("C2", "99.9", "1.5", "modify")]) == []  # best bid is still 100.0
    assert m.feed(t, bids=[order("C2", "99.9", "1.5", "delete")]) == []
    assert t.diag["fifo_anomaly"] == 0


def test_sweep_through_better_level_reaches_us_in_the_same_item():  # scenario 9
    m = Market(bids=[order("B1", "100.0", "1.0"), order("B2", "100.0", "2.0")], asks=[order("A1", "100.2", "3.0")])
    t = m.track(BUY, "99.9")  # quote_deeper: nothing at 99.9 yet
    m.feed(t, bids=[order("C2", "99.9", "2.0", "add")])  # arrives behind us
    reaches = m.feed(t, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                              order("C2", "99.9", "1.2", "modify")])
    (r,) = reaches
    assert (r.volume, r.mechanism, r.ref) == (D("0.8"), Mechanism.LEVEL, (2, 2))


def test_intercept_at_a_worse_touch():  # scenario 10
    m = standard()
    t = m.track(BUY, "100.1")  # improve_quote
    (r,) = m.feed(t, bids=[order("B1", "100.0", "0.7", "modify")])
    assert (r.volume, r.mechanism, r.event_price) == (D("0.3"), Mechanism.INTERCEPT, D("100.0"))
    assert m.feed(t, bids=[order("C1", "99.8", "1.0", "add"), order("C1", "99.8", "0.5", "modify")]) == []  # not touch
    reaches = m.feed(t, bids=[order("B2", "100.0", "2.0", "delete"), order("B1", "100.0", "0.7", "delete")])
    assert [r.volume for r in reaches] == [D("2.0"), D("0.7")]
    assert all(r.mechanism is Mechanism.INTERCEPT for r in reaches)
    (r,) = m.feed(t, bids=[order("C1", "99.8", "0.2", "modify")])  # 99.8 is now the historical touch
    assert (r.volume, r.event_price) == (D("0.3"), D("99.8"))


def test_cross_arrival_buy_and_sell():  # scenario 12
    m = standard()
    buy = m.track(BUY, "100.1")
    sell = m.track(SELL, "100.1")
    out_buy, out_sell = m.feed([buy, sell], asks=[order("A2", "100.1", "0.2", "add")])
    (r,) = out_buy
    assert (r.volume, r.mechanism, r.event_price) == (D("0.2"), Mechanism.CROSS_ARRIVAL, D("100.1"))
    assert out_sell == []  # an ask arriving at our ask price joins our side, behind us
    out_buy, out_sell = m.feed([buy, sell], bids=[order("B7", "100.1", "0.3", "add")])
    assert out_buy == [] and [x.volume for x in out_sell] == [D("0.3")]
    assert m.feed(buy, asks=[order("A3", "100.2", "1.0", "add")]) == []  # does not cross


def test_cross_arrival_with_queue_ahead_is_not_claimed():
    m = standard()
    t = m.track(BUY, "100.0")
    assert m.feed(t, asks=[order("A2", "100.0", "0.5", "add")]) == []
    assert t.diag["cross_with_ahead"] == 1


def test_qty_increase_and_price_move_behind_us_are_not_fills():
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    m.feed(t, bids=[order("B3", "100.0", "4.0", "add")])
    assert m.feed(t, bids=[order("B3", "100.0", "5.0", "modify")]) == []  # increase
    assert m.feed(t, bids=[order("B3", "99.9", "1.0", "modify")]) == []  # moved away with less qty
    assert m.feed(t, bids=[order("ZZ", "100.0", "1.0", "delete")]) == []  # unknown ID: no-op


# --- FIFO priority and requeues -------------------------------------------


def test_price_move_of_order_ahead_loses_priority_and_return_lands_behind():
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B1", "100.0", "1.0", "modify")])  # same qty, same price: nothing changes
    assert list(t.ahead_ids) == ["B1", "B2"]
    m.feed(t, bids=[order("B1", "99.9", "1.0", "modify")])
    assert list(t.ahead_ids) == ["B2"] and t.diag["ahead_requeued"] == 1
    m.feed(t, bids=[order("B1", "100.0", "1.0", "modify")])  # back at our price, but at the back
    assert list(t.ahead_ids) == ["B2"]
    assert [oid for oid, _ in m.book.level_orders("bids", D("100.0"))] == ["B2", "B1"]
    assert_consistent(m, t)


def test_readd_of_known_ahead_id_loses_priority():
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B1", "100.0", "1.0", "add")])
    assert list(t.ahead_ids) == ["B2"] and t.ahead_qty == D("2.0")
    assert [oid for oid, _ in m.book.level_orders("bids", D("100.0"))] == ["B2", "B1"]


def test_modify_of_unknown_id_goes_behind_us():  # scenario 20
    m = standard()
    t = m.track(BUY, "100.0")
    assert m.feed(t, bids=[order("NEW", "100.0", "1.0", "modify")]) == []
    assert "NEW" not in t.ahead_ids and t.ahead_qty == D("3.0")
    assert_consistent(m, t)


# --- invalidation ----------------------------------------------------------


def test_truncated_level_without_events_invalidates_via_apply_info():  # scenario 15
    m = Market(bids=[order("B1", "100.0", "1.0"), order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")],
               depth=2)
    t = m.track(BUY, "99.9")
    events = m.step(bids=[order("B6", "100.1", "1.0", "add")])  # pushes 99.9 out of depth 2
    assert t.on_events(events) == [] and t.invalidated is None  # no event mentions C1
    t.on_truncation(m.book.last_info.truncated)
    assert t.invalidated is EndReason.OUT_OF_DEPTH


def test_truncation_of_a_worse_level_keeps_tracking():
    m = Market(bids=[order("B1", "100.0", "1.0"), order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")],
               depth=2)
    t = m.track(BUY, "100.0")
    m.step(bids=[order("B6", "100.1", "1.0", "add")])
    t.on_truncation(m.book.last_info.truncated)
    assert t.invalidated is None


def test_ahead_order_dropped_by_truncation_invalidates():
    m = Market(bids=[order("B1", "100.0", "1.0"), order("C1", "99.9", "1.0")], asks=[order("A1", "100.2", "3.0")],
               depth=2)
    t = m.track(BUY, "99.9")
    reaches = m.feed(t, bids=[order("C1", "99.9", "0.5", "modify"), order("B6", "100.1", "1.0", "add")])
    assert reaches == [] and t.invalidated is EndReason.OUT_OF_DEPTH
    assert t.advanced_qty == D("1.0")  # recorded, but the order will be censored, not credited


def test_resnapshot_is_invisible_to_the_tracker_so_it_must_be_invalidated():  # scenario 14
    m = standard()
    t = m.track(BUY, "100.0")
    m.feed(t, bids=[order("B1", "100.0", "1.0", "delete")])
    # A mid-stream snapshot reloads the book without emitting events: whatever happened in the
    # hole is unknown, so the engine censors (§7.3). The tracker then ignores everything.
    assert m.book.apply("snapshot", m.sealer.snapshot(bids=[order("B9", "100.0", "5.0")])) is True
    assert m.book.last_events == () and m.book.last_info.book_reset
    t.invalidate(EndReason.CENSORED_RESNAPSHOT)
    before = (dict(t.ahead_ids), t.advanced_qty, t.front_ref)
    assert m.feed(t, bids=[order("B9", "100.0", "5.0", "delete"), order("B2", "100.0", "2.0", "delete")]) == []
    assert (dict(t.ahead_ids), t.advanced_qty, t.front_ref) == before
    assert t.diag["events_after_invalid"] == 2
    t.invalidate(EndReason.CENSORED_DESYNC)
    assert t.invalidated is EndReason.CENSORED_RESNAPSHOT  # first reason wins


def test_audit_reports_divergence():
    m = standard()
    t = m.track(BUY, "100.0")
    t.ahead_ids["B1"] = D("9")
    t.ahead_ids["GHOST"] = D("1")
    problems = t.audit(m.book.level_orders("bids", D("100.0")))
    assert len(problems) == 2
    assert QueueTracker(BUY, D("100.0"), [("B2", D("2.0")), ("B1", D("1.0"))]).audit(
        m.book.level_orders("bids", D("100.0"))
    ) == ["ahead order differs from book priority"]


# --- several hypothetical orders, and the historical book --------------------


def test_multiple_orders_track_independently():
    m = standard()
    early = m.track(BUY, "100.0")
    m.feed(early, bids=[order("B4", "100.0", "4.0", "add")])
    late = m.track(BUY, "100.0")  # inserted after B4, so B4 is ahead of it
    deep = m.track(BUY, "99.9")
    sell = m.track(SELL, "100.2")
    trackers = [early, late, deep, sell]
    assert list(late.ahead_ids) == ["B1", "B2", "B4"] and deep.at_front and list(sell.ahead_ids) == ["A1"]

    m.feed(trackers, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    assert early.at_front and list(late.ahead_ids) == ["B4"]
    out = m.feed(trackers, bids=[order("B4", "100.0", "3.0", "modify")])
    assert [r.volume for r in out[0]] == [D("1.0")]  # B4 is behind `early`: a candidate
    assert out[1] == [] and late.ahead_qty == D("3.0")  # B4 is ahead of `late`: advancement only
    assert out[2] == [] and out[3] == []  # 100.0 is better than 99.9; bids do not touch the ask queue
    for t in trackers:
        assert_consistent(m, t)


def test_tracking_never_changes_the_historical_book():  # invariant I1
    items = [
        dict(bids=[order("B4", "100.0", "4.0", "add")]),
        dict(bids=[order("B1", "100.0", "0.5", "modify")], asks=[order("A2", "100.1", "0.2", "add")]),
        dict(bids=[order("B1", "100.0", "0.5", "delete"), order("B2", "100.0", "2.0", "delete")]),
        dict(bids=[order("B4", "100.0", "3.0", "modify"), order("B2", "99.9", "2.0", "add")]),
    ]
    tracked, plain = standard(), standard()
    trackers = [tracked.track(BUY, "100.0"), tracked.track(BUY, "100.1"), tracked.track(SELL, "100.2")]
    for item in items:
        tracked.feed(trackers, **item)
        plain.step(**item)
        a, b = tracked.book, plain.book
        assert (a.bids, a.asks, a.orders, a.checksum()) == (b.bids, b.asks, b.orders, b.checksum())
    trackers[0].ahead_ids.clear()  # the tracker owns a copy
    assert tracked.book.level_orders("bids", D("100.0")) == plain.book.level_orders("bids", D("100.0"))
