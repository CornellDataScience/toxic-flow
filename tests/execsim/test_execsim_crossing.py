"""Crossed orders: the crossing record (QueueTracker.cross) and the crossing rules of settlement
(execution.CROSS_RULES) through the engine, trade attribution and metrics.

An order is crossed when an opposite-side order comes to rest at or through its price while it is
live. `ignore` keeps counting later candidates, `stop` observes the order up to the crossing item
only, `fill_at_cross` lets a policy that accepts crossing arrivals (optimistic) count them and
makes the others stop. Book for most scenarios (§10.2): bids 100.0: B1 1.0, B2 2.0 | asks 100.2:
A1 3.0; tick 0.1; order qty 0.5; horizon 1 s.
"""

import random
from decimal import Decimal

import pytest
from execsim_helpers import order
from test_execsim_invariants import replay_with_reference
from test_execsim_queue_tracker import standard
from test_execsim_replay import SYM, Tape, at, engine, req
from test_execsim_trades import Public
from test_execsim_trades import run as run_with_trades

from toxicflow.execsim.execution import CROSS_RULES, POLICY_NAMES, REPORTING_CROSS_RULE, FillRecorder, FillStatus, settle
from toxicflow.execsim.metrics import analyze, comparison_rows, format_report
from toxicflow.execsim.models import (CrossEvent, EndReason, Evidence, EvidencePart, FillCandidate, Mechanism,
                                      OrderIntent, PlacementRequest, Side)
from toxicflow.execsim.orders import SimulatedOrder
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy
from toxicflow.execsim.timeutil import MS

D = Decimal
BUY, SELL = Side.BUY, Side.SELL
UNFILLED, PARTIAL, FILLED = FillStatus.UNFILLED, FillStatus.PARTIAL, FillStatus.FILLED
CONS, BAL, OPT = POLICY_NAMES


def replay(tmp_path, tape, schedule, strategies=None, **kw):
    eng = engine(schedule, strategies=strategies, **kw)
    eng.on_reach = FillRecorder()
    return eng.run(tape.write(tmp_path)), eng


def outcome(o, policy, rule, **kw):
    out = settle(o, policy, cross_rule=rule, **kw)
    return out.filled_qty, out.status, out.resolved_by_cross


# --- the crossing record (tracker) ------------------------------------------------------------------


def test_first_crossing_is_recorded_once_for_buy_and_sell():
    m = standard()
    buy, sell = m.track(BUY, "100.1"), m.track(SELL, "100.1")
    assert buy.cross is None and sell.cross is None
    m.feed([buy, sell], asks=[order("A3", "100.2", "1.0", "add")])  # joins the ask side: crosses nobody
    assert buy.cross is None and sell.cross is None
    m.feed([buy, sell], asks=[order("A2", "100.1", "0.2", "add")])  # an ask on the buy's price
    assert buy.cross == CrossEvent((2, 0), 2000, D("100.1"), D("0.2"), D(0), "add")
    assert sell.cross is None  # an ask at our ask price is our own side
    m.feed([buy, sell], asks=[order("A4", "100.0", "0.7", "add")])  # a second, deeper crossing
    assert buy.cross.ref == (2, 0) and buy.diag["crossing"] == 2  # the first one is kept
    m.feed([buy, sell], bids=[order("B7", "100.3", "0.4", "add")])  # a bid through the sell's price
    assert sell.cross == CrossEvent((4, 0), 4000, D("100.3"), D("0.4"), D(0), "add")
    assert buy.diag["crossing"] == 2 and sell.diag["crossing"] == 1


def test_crossing_with_queue_ahead_is_recorded_but_not_a_candidate():
    m = standard()
    t = m.track(BUY, "100.0")  # behind B1 1.0 and B2 2.0
    assert m.feed(t, asks=[order("A2", "100.0", "0.5", "add")]) == []
    assert t.cross == CrossEvent((1, 0), 1000, D("100.0"), D("0.5"), D("3.0"), "add")
    assert t.diag["cross_with_ahead"] == 1 and t.diag["crossing"] == 1
    assert list(t.ahead_ids) == ["B1", "B2"]  # recording a crossing never moves the queue


def test_readd_of_a_resting_crossing_order_is_not_a_new_crossing():
    m = standard()
    t = m.track(BUY, "100.1")
    m.feed(t, asks=[order("A2", "100.0", "0.2", "add")])
    m.feed(t, asks=[order("A2", "100.1", "0.5", "add")])  # same order, still through our price
    assert t.diag["crossing"] == 1 and t.diag["cross_readd"] == 1 and t.cross.ref == (1, 0)


def test_order_repriced_through_us_is_a_crossing_but_never_a_candidate():
    m = standard()
    t = m.track(BUY, "100.1")
    assert m.feed(t, asks=[order("A1", "100.1", "3.0", "modify")]) == []  # A1 moves from 100.2 onto our price
    assert t.cross == CrossEvent((1, 0), 1000, D("100.1"), D("3.0"), D(0), "modify")
    assert m.feed(t, asks=[order("A1", "100.1", "2.0", "modify")]) == []  # already crossing: nothing new
    assert t.diag["crossing"] == 1


# --- settlement rules through the engine ----------------------------------------------------------------


def stale_buy_tape():
    """join_best buy at 100.0: its level is cancelled, an ask arrives ON its price, then a seller
    trades at the lower bid. The later trade is the legacy 'fill'."""
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(20, bids=[order("C1", "99.9", "5.0", "add")])  # the historical touch is now below us
    tape.update(30, asks=[order("A2", "100.0", "0.2", "add")])  # crosses our buy
    tape.update(40, bids=[order("C1", "99.9", "4.6", "modify")])  # 0.4 trades at 99.9, a worse price
    return tape


def stale_sell_tape():
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, asks=[order("A1", "100.2", "3.0", "delete")])
    tape.update(20, asks=[order("C1", "100.3", "5.0", "add")])
    tape.update(30, bids=[order("B9", "100.2", "0.2", "add")])  # a bid on our ask price
    tape.update(40, asks=[order("C1", "100.3", "4.6", "modify")])
    return tape


@pytest.mark.parametrize("side, make", [(BUY, stale_buy_tape), (SELL, stale_sell_tape)])
def test_trade_after_a_crossing_is_a_fill_only_under_ignore(tmp_path, side, make):
    result, _ = replay(tmp_path, make(), [req(5, side)])
    (o,) = result.orders
    assert o.price == (D("100.0") if side is BUY else D("100.2"))
    assert [(c.mechanism, c.evidence, c.qty) for c in o.fills] == [
        (Mechanism.CROSS_ARRIVAL, Evidence.L3_CROSS_ARRIVAL, D("0.2")),
        (Mechanism.INTERCEPT, Evidence.L3_MODIFY_DOWN, D("0.4"))]
    assert o.cross.ts_ns == at(30) and o.cross.ahead_qty == 0 and o.cross.qty == D("0.2")
    assert o.inconsistent_crossing is True  # the market went through us with no reduction reaching us first
    expected = {  # (filled, status, resolved_by_cross) per policy
        "ignore": {CONS: (D(0), UNFILLED, False), BAL: (D("0.4"), PARTIAL, False), OPT: (D("0.5"), FILLED, False)},
        "stop": {CONS: (D(0), UNFILLED, True), BAL: (D(0), UNFILLED, True), OPT: (D(0), UNFILLED, True)},
        "fill_at_cross": {CONS: (D(0), UNFILLED, True), BAL: (D(0), UNFILLED, True),
                          OPT: (D("0.5"), FILLED, False)},
    }
    for rule in CROSS_RULES:
        assert {p: outcome(o, p, rule) for p in POLICY_NAMES} == expected[rule], rule
        assert all(settle(o, p, cross_rule=rule).cross_ts == at(30) for p in POLICY_NAMES)
    opt = settle(o, OPT, cross_rule="fill_at_cross")
    assert [(sf.candidate.mechanism, sf.qty) for sf in opt.fills] == [
        (Mechanism.CROSS_ARRIVAL, D("0.2")), (Mechanism.INTERCEPT, D("0.3"))]  # the arrival's own qty, no more
    assert opt.first_fill_ts == at(30) and opt.complete_ts == at(40)
    assert settle(o, BAL).filled_qty == D("0.4")  # the default is the legacy rule
    with pytest.raises(ValueError):
        settle(o, BAL, cross_rule="first")


def test_fill_before_the_crossing_is_kept_under_every_rule(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])  # behind us
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(30, bids=[order("B3", "100.0", "3.8", "modify")])  # 0.2 reaches us: a fill before any crossing
    tape.update(40, bids=[order("B3", "100.0", "3.8", "delete")])
    tape.update(50, asks=[order("A2", "100.0", "9.0", "add")])  # crossed
    tape.update(60, bids=[order("C1", "99.9", "1.0", "add")])
    tape.update(70, bids=[order("C1", "99.9", "0.5", "modify")])  # after the crossing
    result, _ = replay(tmp_path, tape, [req(5)])
    (o,) = result.orders
    assert o.inconsistent_crossing is False  # reductions reached us before the crossing
    assert outcome(o, BAL, "ignore") == (D("0.5"), FILLED, False)
    for rule in ("stop", "fill_at_cross"):
        out = settle(o, BAL, cross_rule=rule)
        assert (out.filled_qty, out.status, out.resolved_by_cross) == (D("0.2"), PARTIAL, True)
        assert out.first_fill_ts == at(30) and out.complete_ts is None
    # optimistic also counts the delete (rank 2) before the crossing: complete, so nothing left to resolve
    assert outcome(o, OPT, "stop") == (D("0.5"), FILLED, False)


def test_crossing_with_queue_ahead_never_fills_by_itself(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])  # behind us
    tape.update(20, asks=[order("A2", "100.0", "0.5", "add")])  # crosses while B1 and B2 are still ahead
    tape.update(30, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(40, bids=[order("B3", "100.0", "3.8", "modify")])  # 0.2 behind us at the touch
    result, _ = replay(tmp_path, tape, [req(5)])
    (o,) = result.orders
    assert o.cross.ahead_qty == D("3.0") and o.diag["cross_with_ahead"] == 1
    assert [c.mechanism for c in o.fills] == [Mechanism.LEVEL]  # the crossing produced no candidate
    assert outcome(o, BAL, "ignore") == (D("0.2"), PARTIAL, False)
    assert outcome(o, BAL, "stop") == (D(0), UNFILLED, True)
    assert outcome(o, OPT, "stop") == (D(0), UNFILLED, True)
    # fill_at_cross invents nothing at the crossing: optimistic only has the later reduction, as in ignore
    assert outcome(o, OPT, "fill_at_cross") == (D("0.2"), PARTIAL, False)
    assert outcome(o, BAL, "fill_at_cross") == (D(0), UNFILLED, True)


def test_sweep_that_rests_its_remainder_fills_in_the_crossing_item(tmp_path):
    # one aggressive sell: takes B1 and B2, 0.3 of B3 (behind us), and rests the remainder on our price
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete"),
                          order("B3", "100.0", "3.7", "modify")],
                asks=[order("A2", "100.0", "0.5", "add")])
    tape.update(30, bids=[order("B3", "100.0", "3.0", "modify")])  # a later item: after the crossing
    result, _ = replay(tmp_path, tape, [req(5)])
    (o,) = result.orders
    assert [(c.mechanism, c.qty, c.ts_ns) for c in o.fills] == [
        (Mechanism.LEVEL, D("0.3"), at(20)), (Mechanism.CROSS_ARRIVAL, D("0.5"), at(20)),
        (Mechanism.LEVEL, D("0.7"), at(30))]
    assert o.cross.ts_ns == at(20) and o.inconsistent_crossing is False
    assert outcome(o, BAL, "ignore") == (D("0.5"), FILLED, False)
    out = settle(o, BAL, cross_rule="stop")  # the sweep's own reduction counts; the next item does not
    assert (out.filled_qty, out.status, out.resolved_by_cross) == (D("0.3"), PARTIAL, True)
    assert [sf.candidate.ts_ns for sf in out.fills] == [at(20)]
    assert outcome(o, OPT, "stop") == (D("0.3"), PARTIAL, True)  # never the arrival itself
    assert outcome(o, OPT, "fill_at_cross") == (D("0.5"), FILLED, False)


def test_crossing_at_expiry_does_not_count_and_one_before_does(tmp_path):
    for ms, crossed in ((1005, False), (1004, True)):  # the order is placed at 5 ms with a 1 s horizon
        tape = Tape()
        tape.snapshot(0)
        tape.update(10, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
        tape.update(ms, asks=[order("A2", "100.0", "0.2", "add")])
        tape.update(2000, asks=[order("A2", "100.0", "0.2", "delete")])
        result, _ = replay(tmp_path / str(ms), tape, [req(5)])
        (o,) = result.orders
        assert o.end_reason is EndReason.HORIZON and o.expiry_ts == at(1005)
        assert (o.cross is not None) is crossed  # an item at exactly the expiry time is after the order's life
        assert settle(o, BAL, cross_rule="stop").resolved_by_cross is crossed
        if crossed:  # a shorter analysis window that ends at the crossing does not contain it
            out = settle(o, BAL, until_ns=at(1004), cross_rule="stop")
            assert out.cross_ts is None and not out.resolved_by_cross


def test_several_partial_crossing_arrivals(tmp_path):
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, asks=[order("A2", "100.1", "0.2", "add")])  # on improve_quote's 100.1 buy
    tape.update(20, asks=[order("A3", "100.0", "0.1", "add")])
    tape.update(30, asks=[order("A4", "100.1", "0.4", "add")])
    result, _ = replay(tmp_path, tape, [req(5)], strategies=[ImproveQuoteStrategy()])
    (o,) = result.orders
    assert o.price == D("100.1") and o.diag["crossing"] == 3
    assert (o.cross.ts_ns, o.cross.qty, o.cross.ref) == (at(10), D("0.2"), o.fills[0].event_ref)
    for rule in ("ignore", "fill_at_cross"):  # each arrival counts for its own quantity, clipped once
        out = settle(o, OPT, cross_rule=rule)
        assert [sf.qty for sf in out.fills] == [D("0.2"), D("0.1"), D("0.2")] and out.status is FILLED
        assert (out.first_fill_ts, out.complete_ts) == (at(10), at(30))
        assert len({sf.candidate.event_ref for sf in out.fills}) == 3
        assert outcome(o, BAL, rule)[0] == 0  # rank 1 is below balanced
    assert outcome(o, OPT, "stop") == (D(0), UNFILLED, True)
    assert outcome(o, OPT, "ignore", until_ns=at(25)) == (D("0.3"), PARTIAL, False)  # partial at a shorter horizon


def test_censored_order_keeps_its_crossing(tmp_path):
    tape = stale_buy_tape()
    tape.update(50, bids=[order("C1", "99.9", "4.0", "modify")], corrupt=True)  # checksum failure: desync
    result, _ = replay(tmp_path, tape, [req(5)])
    (o,) = result.orders
    assert o.end_reason is EndReason.CENSORED_DESYNC and o.censored and o.cross.ts_ns == at(30)
    assert len(o.fills) == 2  # the corrupt item's events never reached the tracker
    assert outcome(o, BAL, "stop") == (D(0), UNFILLED, True)
    assert outcome(o, BAL, "ignore") == (D("0.4"), PARTIAL, False)


def test_replay_with_crossed_orders_leaves_the_book_untouched(tmp_path):
    tape = stale_buy_tape()
    _, with_orders = replay(tmp_path / "a", tape, [req(5), req(5, SELL)],
                            strategies=[JoinBestStrategy(), ImproveQuoteStrategy()])
    _, without = replay(tmp_path / "b", tape, [])
    a, b = with_orders.state[SYM].book, without.state[SYM].book
    assert (a.bids, a.asks, a.orders, a.checksum()) == (b.bids, b.asks, b.orders, b.checksum())


# --- with trade evidence -------------------------------------------------------------------------------


def test_trade_confirmed_fill_after_a_crossing_follows_the_rule(tmp_path):
    pub = Public()
    pub.trade(at(40), "sell", "99.9", "0.4")  # the print behind the reduction at 99.9
    result, att, _ = run_with_trades(tmp_path, stale_buy_tape(), pub, schedule=[req(5)])
    (o,) = result.orders
    late = o.fills[1]
    assert [(p.evidence, p.qty) for p in late.attribution] == [(Evidence.TRADE_CONFIRMED, D("0.4"))]
    assert o.fills[0].attribution == (EvidencePart(D("0.2"), Evidence.L3_CROSS_ARRIVAL),)
    for p in (CONS, BAL):  # a confirmed trade, but after our order stopped being a resting order
        assert outcome(o, p, "ignore", with_trades=True) == (D("0.4"), PARTIAL, False)
        assert outcome(o, p, "stop", with_trades=True) == (D(0), UNFILLED, True)
        assert outcome(o, p, "fill_at_cross", with_trades=True) == (D(0), UNFILLED, True)
    assert outcome(o, OPT, "fill_at_cross", with_trades=True) == (D("0.5"), FILLED, False)
    # the rule never rewrites the evidence: one attribution, the print used once
    assert sum(att.trade_used.values()) == D("0.4")
    assert [c.attribution for c in o.fills] == [o.fills[0].attribution, late.attribution]


# --- ordering and accounting properties -------------------------------------------------------------------


def check_rules(o, with_trades, untils):
    """Per rule: conservative <= balanced <= optimistic, bounded by the order quantity, monotone in
    the horizon, and no candidate counted twice. Across rules: stop <= fill_at_cross <= ignore."""
    for until in untils:
        got = {rule: [settle(o, p, until_ns=until, with_trades=with_trades, cross_rule=rule) for p in POLICY_NAMES]
               for rule in CROSS_RULES}
        for rule, outs in got.items():
            qty = [out.filled_qty for out in outs]
            assert D(0) <= qty[0] <= qty[1] <= qty[2] <= o.qty, (rule, qty)
            for out in outs:
                refs = [sf.candidate.event_ref for sf in out.fills]
                assert len(refs) == len(set(refs)) and sum((sf.qty for sf in out.fills), D(0)) == out.filled_qty
                assert all(sf.qty > 0 and sf.qty <= sf.candidate.qty for sf in out.fills)
                assert all(sum((part.qty for part in sf.by_evidence), D(0)) == sf.qty for sf in out.fills)
                if out.resolved_by_cross:
                    assert out.cross_ts is not None and out.filled_qty < o.qty
                    assert all(sf.candidate.ts_ns <= out.cross_ts for sf in out.fills)
        for i in range(len(POLICY_NAMES)):
            stop, at_cross, ignore = (got[rule][i].filled_qty for rule in ("stop", "fill_at_cross", "ignore"))
            assert stop <= at_cross <= ignore
            assert not got["ignore"][i].resolved_by_cross
    for rule in CROSS_RULES:
        for p in POLICY_NAMES:
            series = [settle(o, p, until_ns=u, with_trades=with_trades, cross_rule=rule).filled_qty for u in untils]
            assert series == sorted(series)  # a longer window never un-fills


@pytest.mark.parametrize("seed", range(30))
def test_policy_ordering_under_every_rule_on_random_streams(seed, tmp_path, monkeypatch):
    _, result, cfg, *_ = replay_with_reference(seed, tmp_path, monkeypatch)
    placed = [o for o in result.orders if o.inserted_ts is not None]
    for o in placed:
        steps = sorted({o.inserted_ts, o.expiry_ts, *(c.ts_ns for c in o.fills), *(c.ts_ns + 1 for c in o.fills)})
        check_rules(o, False, steps)
        if o.cross is not None:
            assert o.inserted_ts <= o.cross.ts_ns < o.expiry_ts and o.cross.ts_ns <= o.end_ts
            assert o.diag["crossing"] >= 1
            assert all(c.ts_ns >= o.cross.ts_ns for c in o.fills if c.mechanism is Mechanism.CROSS_ARRIVAL)


def test_random_streams_contain_crossings(tmp_path_factory, monkeypatch):
    crossed = late = 0
    for seed in range(30):
        _, result, *_ = replay_with_reference(seed, tmp_path_factory.mktemp(f"s{seed}"), monkeypatch)
        for o in result.orders:
            if o.cross is not None:
                crossed += 1
                late += any(c.ts_ns > o.cross.ts_ns for c in o.fills)
    assert crossed > 20 and late > 5  # the property test above is not vacuous


@pytest.mark.parametrize("seed", range(40))
def test_policy_ordering_with_random_trade_slices(seed):
    """Hand-built orders with random candidates, trade slices and a crossing at a random time."""
    rng = random.Random(seed)
    request = PlacementRequest("r", SYM, BUY, 0)
    o = SimulatedOrder.new(request, "s", OrderIntent(BUY, D("100.0")), D(rng.choice(["0.1", "0.5", "2"])), 0)
    o.start(inserted_ts=0, ahead=(), rank=1, touch_qty=D(1), horizon_ns=1_000)
    times = sorted(rng.randrange(0, 1_000) for _ in range(rng.randrange(0, 12)))
    cross_ts = rng.choice(times) if times and rng.random() < 0.8 else None
    for i, ts in enumerate(times):
        is_cross = cross_ts is not None and ts >= cross_ts and (ts == cross_ts or rng.random() < 0.4)
        qty = D(rng.randrange(1, 30)) / 100
        if is_cross:
            mech, label, parts = Mechanism.CROSS_ARRIVAL, Evidence.L3_CROSS_ARRIVAL, (
                EvidencePart(qty, Evidence.L3_CROSS_ARRIVAL),)
        else:
            mech = rng.choice([Mechanism.LEVEL, Mechanism.INTERCEPT])
            label = rng.choice([Evidence.L3_MODIFY_DOWN, Evidence.L3_DELETE_AMBIGUOUS])
            cut = (D(rng.randrange(0, 101)) / 100 * qty).quantize(D("0.01"))
            kinds = rng.sample([Evidence.TRADE_CONFIRMED, Evidence.TRADE_PLAUSIBLE, Evidence.TRADE_ABSENT, label], 2)
            parts = tuple(EvidencePart(q, e) for q, e in zip((cut, qty - cut), kinds) if q > 0)
        o.add_fill(FillCandidate(ts, o.price, qty, mech, label, (i, 0), attribution=parts))
    if cross_ts is not None:
        first = next(c for c in o.fills if c.ts_ns == cross_ts)
        o.tracker.cross = CrossEvent(first.event_ref, cross_ts, o.price, first.qty, D(0), "add")
    untils = sorted({0, 1_000, *times, *(t + 1 for t in times)})
    check_rules(o, False, untils)
    check_rules(o, True, untils)


# --- metrics -----------------------------------------------------------------------------------------------


def three_order_tape():
    """Three join_best buys: one filled before any crossing, one crossed and then traded through
    (the legacy fill), one that simply expires."""
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])
    tape.update(20, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "2.0", "delete")])
    tape.update(40, bids=[order("B3", "100.0", "3.4", "modify")])  # 0.6: order 1 fills
    # order 2 joins 100.0 behind B3 at 2005
    tape.update(2010, bids=[order("B3", "100.0", "3.4", "delete")])
    tape.update(2020, bids=[order("C1", "99.9", "5.0", "add")])
    tape.update(2030, asks=[order("A2", "100.0", "0.2", "add")])  # order 2 is crossed
    tape.update(2040, bids=[order("C1", "99.9", "4.6", "modify")])  # 0.4 at a worse price
    tape.update(2900, asks=[order("A2", "100.0", "0.2", "delete")])
    # order 3 joins 99.9 behind C1 at 4005 and nothing happens
    tape.update(6000, asks=[order("A9", "100.9", "1.0", "add")])
    tape.update(6500, asks=[order("A9", "100.9", "1.0", "delete")])
    return tape


def test_metrics_report_crossed_orders_separately(tmp_path):
    result, eng = replay(tmp_path, three_order_tape(), [req(5), req(2005), req(4005)])
    cfg = eng.cfg
    reports = {rule: analyze(result, cfg, cross_rule=rule) for rule in CROSS_RULES}
    assert analyze(result, cfg).cross_rule == "ignore" and REPORTING_CROSS_RULE == "stop"
    for rep in reports.values():  # placement and crossing do not depend on the rule
        lc = rep.lifecycle["join_best"]
        assert (lc.placed, lc.expired, lc.censored, lc.crossed, lc.inconsistent_crossings) == (3, 3, 0, 1, 1)

    ig = reports["ignore"].summary[("join_best", BAL)]
    assert (ig.filled_any, ig.unfilled_observed, ig.crossed_unfilled, ig.resolved_by_cross) == (2, 1, 0, 0)
    assert (ig.fill_given_placed.num, ig.fill_given_placed.den) == (2, 3)
    assert (ig.fills_after_cross, ig.legacy_fills_after_cross, ig.legacy_first_fill_after_cross) == (1, 1, 1)

    st = reports["stop"].summary[("join_best", BAL)]
    assert (st.filled_any, st.unfilled_observed, st.crossed_unfilled, st.resolved_by_cross) == (1, 1, 1, 1)
    assert st.censored_unfilled == 0  # a crossing is not data censoring
    assert (st.fill_given_placed.num, st.fill_given_placed.den) == (1, 2)  # the unknown outcome is not a non-fill
    assert (st.fill_given_placed_lower.num, st.fill_given_placed_lower.den) == (1, 3)
    assert (st.fill_given_placed_upper.num, st.fill_given_placed_upper.den) == (2, 3)
    assert (st.fill_per_opportunity.num, st.fill_per_opportunity.den) == (1, 2)
    assert (st.fills_after_cross, st.n_fills) == (0, 1)
    assert (st.legacy_filled_any, st.legacy_fills_after_cross, st.legacy_first_fill_after_cross) == (2, 1, 1)
    # Kaplan-Meier: one event at 35 ms, the crossed order censored at its crossing (25 ms), one at the horizon
    assert (st.km_first_fill.n, st.km_first_fill.n_events, st.km_first_fill.n_censored) == (3, 1, 2)
    assert st.km_first_fill.curve == ((35 * MS, 0.5),)

    fc = reports["fill_at_cross"]
    assert fc.summary[("join_best", BAL)] == st.__class__(**{
        **{f: getattr(st, f) for f in st.__dataclass_fields__}})  # balanced stops exactly as under `stop`
    opt = fc.summary[("join_best", OPT)]
    assert (opt.filled_any, opt.crossed_unfilled, opt.fills_by_mechanism) == (
        2, 0, {"cross_arrival": 1, "intercept": 1, "level": 1})
    assert reports["stop"].summary[("join_best", OPT)].filled_any == 1

    # the settled fills carry the flag, and the text report states the rule and what it changed
    assert [f.after_cross for f in reports["ignore"].fills if f.policy == BAL] == [False, True]
    text = format_report(reports["stop"])
    assert "cross rule stop" in text and "unknown at crossing 1 (unfilled 1)" in text
    assert "'ignore' would report 2 filled, 1 first filled after the crossing" in text
    (row,) = comparison_rows(reports["stop"], BAL)
    assert (row["cross_rule"], row["crossed"], row["crossed_unfilled"], row["legacy_filled_any"]) == ("stop", 1, 1, 2)
    with pytest.raises(ValueError):
        analyze(result, cfg, cross_rule="later")
