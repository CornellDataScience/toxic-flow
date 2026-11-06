"""Evidence and settlement (execution.py), mid sampling and markouts (markouts.py) and strategy
metrics (metrics.py), on hand-built results with known answers and on synthetic tapes (design §6.4 to
§6.7, §11.4, §11.5)."""

import copy
from decimal import Decimal

import pytest
from execsim_helpers import order
from test_execsim_replay import BIDS, SYM, Tape, at, engine, req

from toxicflow.execsim.execution import FillRecorder, FillStatus, l3_evidence, settle
from toxicflow.execsim.markouts import MidSample, sample_mids, signed_markout_bps
from toxicflow.execsim.metrics import (COVERAGE, DATA_GATE, ELIGIBLE, INSERT_SKIPPED, PLACED, SKIPPED, Dist, Rate,
                                       analyze, comparison_rows, evaluate, format_report, kaplan_meier,
                                       required_mid_times)
from toxicflow.execsim.models import (BookView, EndReason, Evidence, FillCandidate, LevelSummary, Mechanism,
                                      OrderIntent, PlacementRequest, ReachEvent, Side, SkipReason)
from toxicflow.execsim.orders import SimulatedOrder
from toxicflow.execsim.results import QualityEvent, ReachRecord, ReplayResult, SkipRecord
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy
from toxicflow.execsim.timeutil import MS, S

D = Decimal
BUY, SELL = Side.BUY, Side.SELL
H = 10 * S
END = 10_000 * S  # far beyond every window: the data covers everything unless a test says otherwise
MOD, DEL, CROSS = Evidence.L3_MODIFY_DOWN, Evidence.L3_DELETE_AMBIGUOUS, Evidence.L3_CROSS_ARRIVAL
LEVEL, INTERCEPT, ARRIVAL = Mechanism.LEVEL, Mechanism.INTERCEPT, Mechanism.CROSS_ARRIVAL


# --- builders for hand-made results ----------------------------------------------------------------------


def cfg(horizon_ns=H, latency_ns=0):
    from toxicflow.execsim.config import SimConfig
    return SimConfig(symbols=(SYM,), order_qty={SYM: "1"}, horizon_ns=horizon_ns, latency_ns=latency_ns)


def rq(i, side=BUY):
    """Request i at i * 100 s (well apart, so windows never overlap)."""
    return PlacementRequest(f"r{i}{side.value[0]}", SYM, side, i * 100 * S)


def view(r, bid="100.0", ask="100.2"):
    return BookView(SYM, r.ts_ns, D("0.1"), (LevelSummary(D(bid), D("3"), 2),), (LevelSummary(D(ask), D("3"), 1),))


class Result:
    """Accumulates a ReplayResult by hand."""

    def __init__(self):
        self.orders, self.skips, self.reaches, self.views = [], [], [], {}
        self.quality = [QualityEvent(END, SYM, "END_OF_DATA", None, None, {}, None)]

    def request(self, r, gate=None, **view_kw):
        if gate is None:
            self.views[r.request_id] = view(r, **view_kw)
        return r

    def gated(self, r, reason, strategies=("join_best",)):
        for s in strategies:
            self.skips.append(SkipRecord(r, s, reason))

    def skip(self, r, strategy, reason):
        self.skips.append(SkipRecord(r, strategy, reason))

    def place(self, r, strategy="join_best", price=None, qty="1", fills=(), end=EndReason.HORIZON, end_ts=None,
              insert_skip=None, ahead=(("B1", "1.0"), ("B2", "2.0")), rank=1, touch="3.0", horizon=H):
        """fills: (offset_ns after insertion, volume, mechanism, evidence[, ahead_qty])."""
        price = D(price or ("100.0" if r.side is BUY else "100.2"))
        o = SimulatedOrder.new(r, strategy, OrderIntent(r.side, price), D(qty), r.ts_ns)
        self.orders.append(o)
        if insert_skip is not None:
            o.skip(insert_skip)
            return o
        o.start(r.ts_ns, [(oid, D(q)) for oid, q in ahead], rank, D(touch), horizon)
        for k, (dt, vol, mech, ev, *rest) in enumerate(fills):
            ref = (len(self.reaches), k)
            o.add_fill(FillCandidate(r.ts_ns + dt, price, D(vol), mech, ev, ref))
            reach = ReachEvent(ref, r.ts_ns + dt, price, D(vol), mech)
            self.reaches.append(ReachRecord(o.order_id, strategy, SYM, reach, D(rest[0] if rest else 0), True))
        o.finish(end, o.expiry_ts if end_ts is None and end is EndReason.HORIZON else r.ts_ns + (end_ts or 0))
        return o

    def gap(self, ts_ns, kind="DESYNC"):
        self.quality.append(QualityEvent(ts_ns, SYM, kind, None, None, {}, 1))

    def result(self):
        return ReplayResult(self.orders, self.skips, self.reaches, self.quality, self.views, {})


def mids(*pairs):
    """{(SYM, ts): MidSample} from (ts, mid or a missing reason)."""
    out = {}
    for ts, m in pairs:
        missing = m if isinstance(m, str) and not m[0].isdigit() else None
        out[(SYM, ts)] = MidSample(SYM, ts, None if missing else D(m), missing)
    return out


def rate(r: Rate):
    return (r.num, r.den)


# --- execution: evidence and settlement ---------------------------------------------------------------------


def test_l3_evidence_ladder():
    assert l3_evidence(ARRIVAL, "add") is CROSS
    assert l3_evidence(LEVEL, "delete") is DEL and l3_evidence(INTERCEPT, "delete") is DEL
    assert l3_evidence(LEVEL, "modify") is MOD and l3_evidence(INTERCEPT, "modify") is MOD
    with pytest.raises(ValueError):
        l3_evidence(LEVEL, "add")


def test_settle_worked_example_partial_then_full():  # §6.4 steps 5 and 6
    res = Result()
    o = res.place(rq(1), qty="0.5", fills=[(5, "0.2", LEVEL, MOD), (6, "0.6", LEVEL, MOD), (7, "1.0", LEVEL, DEL)])
    bal = settle(o, "balanced")
    assert (bal.filled_qty, bal.status, bal.first_fill_ts, bal.complete_ts) == (D("0.5"), FillStatus.FILLED,
                                                                                o.inserted_ts + 5, o.inserted_ts + 6)
    assert [f.qty for f in bal.fills] == [D("0.2"), D("0.3")]  # clipped; the extra 0.3 and the delete unused
    assert settle(o, "conservative").status is FillStatus.UNFILLED  # L3-only: nothing qualifies
    part = settle(o, "balanced", until_ns=o.inserted_ts + 6)  # a shorter horizon: ts >= cutoff never counts
    assert (part.filled_qty, part.status, part.complete_ts) == (D("0.2"), FillStatus.PARTIAL, None)


def test_settle_ambiguous_delete_counts_only_for_optimistic():  # scenario 6
    res = Result()
    o = res.place(rq(1), qty="0.5", fills=[(5, "4.0", LEVEL, DEL)])
    assert settle(o, "optimistic").status is FillStatus.FILLED
    assert settle(o, "balanced").status is settle(o, "conservative").status is FillStatus.UNFILLED


def test_settle_skipped_order_is_unfilled():
    res = Result()
    o = res.place(rq(1), insert_skip=SkipReason.OUT_OF_DEPTH)
    out = settle(o, "optimistic")
    assert (out.filled_qty, out.status, out.fills) == (0, FillStatus.UNFILLED, ())


def test_fill_recorder_on_the_engine(tmp_path):  # §6.4 worked example through the replay engine
    tape = Tape()
    tape.snapshot(0)
    tape.update(10, bids=[order("B3", "100.0", "4.0", "add")])
    tape.update(20, bids=[order("B1", "100.0", "0.6", "modify")])
    tape.update(30, bids=[order("B2", "100.0", "2.0", "delete")])
    tape.update(40, bids=[order("B1", "100.0", "0.6", "delete")])
    tape.update(50, bids=[order("B3", "100.0", "3.8", "modify")])
    tape.update(60, bids=[order("B3", "100.0", "3.2", "modify")])
    tape.update(70, bids=[order("B3", "100.0", "3.2", "delete")])
    eng = engine([req(5)])
    eng.on_reach = FillRecorder()
    result = eng.run(tape.write(tmp_path))
    (o,) = result.orders
    assert [(c.ts_ns, c.qty, c.mechanism, c.evidence, c.price) for c in o.fills] == [
        (at(50), D("0.2"), LEVEL, MOD, D("100.0")), (at(60), D("0.6"), LEVEL, MOD, D("100.0")),
        (at(70), D("3.2"), LEVEL, DEL, D("100.0"))]
    assert [c.event_ref for c in o.fills] == [r.reach.ref for r in result.reaches]
    bal = settle(o, "balanced")
    assert (bal.status, bal.first_fill_ts, bal.complete_ts) == (FillStatus.FILLED, at(50), at(60))
    assert o.end_reason is EndReason.CENSORED_END_OF_DATA  # the order life cycle is unchanged


# --- Kaplan-Meier and small types ---------------------------------------------------------------------------


def test_kaplan_meier_by_hand():
    km = kaplan_meier([(1, True), (2, False), (3, True), (4, False)])
    # S(1) = 3/4; at t=3 two at risk: S = 3/4 * 1/2 = 3/8
    assert (km.n, km.n_events, km.n_censored, km.median_ns) == (4, 2, 2, 3)
    assert km.curve == ((1, 0.75), (3, 0.375)) and km.fill_prob_at_horizon == pytest.approx(0.625)
    ties = kaplan_meier([(5, True), (5, False), (5, True), (9, False)])  # events before censorings at a tie
    assert ties.curve == ((5, 0.5),) and ties.median_ns == 5
    none = kaplan_meier([(10, False)] * 3)
    assert (none.median_ns, none.fill_prob_at_horizon, none.curve) == (None, 0.0, ())
    assert kaplan_meier([]).fill_prob_at_horizon is None


def test_rate_and_dist():
    assert Rate(0, 0).value is None and str(Rate(0, 0)) == "n/a (0/0)"
    assert Rate(1, 4).value == 0.25 and Rate(0, 3).value == 0.0
    assert Dist.of([]) == Dist(0, None, None, None, None)
    assert Dist.of([3, 1, 2, 10]) == Dist(4, D(4), D("2.5"), D(1), D(10))


# --- denominators and outcomes -----------------------------------------------------------------------------


def test_no_requests():
    rep = analyze(Result().result(), cfg(), strategies=["join_best"])
    lc, s = rep.lifecycle["join_best"], rep.summary[("join_best", "balanced")]
    assert rep.request_counts["total"] == rep.request_counts["eligible"] == 0
    assert lc.placement_rate.value is None and s.fill_given_placed.value is None
    assert s.fill_per_opportunity.value is None and s.km_first_fill.n == 0 and s.markouts[S].n_fills == 0


def test_data_gated_requests_are_not_opportunities():
    res = Result()
    res.gated(rq(1), SkipReason.UNSYNCED)
    res.gated(rq(2), SkipReason.NO_DATA)
    rep = analyze(res.result(), cfg())
    assert rep.request_counts["data_gate"] == {"NO_DATA": 1, "UNSYNCED": 1} and rep.request_counts["eligible"] == 0
    assert [r.status for r in rep.requests] == [DATA_GATE, DATA_GATE]
    assert rep.lifecycle["join_best"].eligible == 0 and rep.lifecycle["join_best"].strategy_skips == 0


def test_all_strategy_skips():
    res = Result()
    for i in range(3):
        res.skip(res.request(rq(i)), "improve_quote", SkipReason.SPREAD_TOO_NARROW)
    rep = analyze(res.result(), cfg())
    lc, s = rep.lifecycle["improve_quote"], rep.summary[("improve_quote", "optimistic")]
    assert (lc.eligible, lc.strategy_skips, lc.placed) == (3, 3, 0)
    assert lc.strategy_skip_reasons == {"SPREAD_TOO_NARROW": 3} and rate(lc.placement_rate) == (0, 3)
    assert s.fill_given_placed.value is None  # 0/0: undefined, not zero
    assert rate(s.fill_per_opportunity) == (0, 3)  # a skip is an opportunity with no fill


def test_all_placed_unfilled():
    res = Result()
    for i in range(4):
        res.place(res.request(rq(i)))
    rep = analyze(res.result(), cfg())
    s = rep.summary[("join_best", "optimistic")]
    assert (s.filled_any, s.unfilled_observed, s.censored_unfilled) == (0, 4, 0)
    assert rate(s.fill_given_placed) == (0, 4) and rep.lifecycle["join_best"].expired == 4
    assert (s.km_first_fill.n_events, s.km_first_fill.fill_prob_at_horizon, s.km_first_fill.median_ns) == (0, 0.0, None)


def test_all_filled_with_exact_times():
    res = Result()
    for i, dt in enumerate((5 * MS, 1 * S, 3 * S)):
        res.place(res.request(rq(i)), fills=[(dt, "2.0", LEVEL, MOD)])
    rep = analyze(res.result(), cfg())
    s = rep.summary[("join_best", "balanced")]
    assert (s.filled_any, s.filled_full, s.partial) == (3, 3, 0) and rate(s.fill_given_placed) == (3, 3)
    assert s.time_to_first_fill_ns == Dist(3, D(4_005 * MS) / 3, D(S), D(5 * MS), D(3 * S))
    assert s.time_to_complete_fill_ns.median == D(S) and s.km_first_fill.median_ns == S
    assert s.km_first_fill.fill_prob_at_horizon == 1.0
    assert rep.summary[("join_best", "conservative")].filled_any == 0


def test_mixed_outcomes_and_every_denominator():
    res = Result()
    res.place(res.request(rq(0)), fills=[(2 * S, "1.0", LEVEL, MOD)])  # filled, observed
    res.place(res.request(rq(1)))  # expired unfilled
    res.place(res.request(rq(2)), end=EndReason.OUT_OF_DEPTH, end_ts=4 * S)  # censored, no fill: unknown
    res.place(res.request(rq(3)), qty="2", fills=[(1 * S, "0.5", INTERCEPT, MOD, "0.7")],
              end=EndReason.OUT_OF_DEPTH, end_ts=6 * S)  # censored after a partial fill: filled_any
    res.place(res.request(rq(4)), insert_skip=SkipReason.CROSSED_AT_INSERT)  # refused at insertion
    res.skip(res.request(rq(5)), "join_best", SkipReason.NO_TOUCH)
    rep = analyze(res.result(), cfg())
    lc, s = rep.lifecycle["join_best"], rep.summary[("join_best", "balanced")]
    assert (lc.eligible, lc.strategy_skips, lc.orders_created, lc.insertion_skips, lc.placed) == (6, 1, 5, 1, 4)
    assert (lc.expired, lc.censored, lc.censored_reasons) == (3 - 1, 2, {"OUT_OF_DEPTH": 2})
    assert lc.insertion_skip_reasons == {"CROSSED_AT_INSERT": 1}
    assert rate(lc.placement_rate) == (4, 6) and rate(lc.censored_rate) == (2, 4)
    assert (s.filled_any, s.filled_full, s.partial, s.unfilled_observed, s.censored_unfilled, s.full_undetermined) == (
        2, 1, 1, 1, 1, 2)  # both censored orders: neither is known to have completed
    assert rate(s.fill_given_placed) == (2, 3)  # the censored unfilled order is excluded, not a non-fill
    assert rate(s.fill_given_placed_lower) == (2, 4) and rate(s.fill_given_placed_upper) == (3, 4)
    assert rate(s.fill_per_opportunity) == (2, 5)
    assert rate(s.full_fill_given_placed) == (1, 2) and rate(s.full_fill_per_opportunity) == (1, 4)
    # KM: events at 1 s and 2 s, censored at 4 s and at H
    assert s.km_first_fill.curve == ((S, 0.75), (2 * S, 0.5)) and s.km_first_fill.median_ns == 2 * S
    assert s.ahead_qty_at_fill == Dist.of([0, D("0.7")]) and s.fills_by_mechanism == {"intercept": 1, "level": 1}
    stages = [(o.request.request_id, o.stage) for o in rep.orders]
    assert stages[4] == ("r4b", INSERT_SKIPPED) and stages[5] == ("r5b", SKIPPED)
    assert sum(o.stage == PLACED for o in rep.orders) == 4


def test_multiple_fills_per_order_are_settled_once():
    res = Result()
    res.place(res.request(rq(0)), qty="1", fills=[(1 * S, "0.4", LEVEL, MOD), (2 * S, "0.4", ARRIVAL, CROSS),
                                                  (3 * S, "0.4", LEVEL, MOD), (4 * S, "0.4", LEVEL, MOD)])
    rep = analyze(res.result(), cfg())
    bal = [f for f in rep.fills if f.policy == "balanced"]
    opt = [f for f in rep.fills if f.policy == "optimistic"]
    assert [(f.since_insert_ns, f.qty) for f in bal] == [(S, D("0.4")), (3 * S, D("0.4")), (4 * S, D("0.2"))]
    assert [(f.since_insert_ns, f.qty) for f in opt] == [(S, D("0.4")), (2 * S, D("0.4")), (3 * S, D("0.2"))]
    for p in ("balanced", "optimistic"):
        s = rep.summary[("join_best", p)]
        assert (s.filled_full, s.n_fills) == (1, 3)
    assert rep.summary[("join_best", "optimistic")].time_to_complete_fill_ns.median == D(3 * S)


def test_policies_are_ordered():  # I6 on the analysis side
    res = Result()
    res.place(res.request(rq(0)), fills=[(1, "0.3", ARRIVAL, CROSS), (2, "0.3", LEVEL, DEL), (3, "0.3", LEVEL, MOD)])
    rep = analyze(res.result(), cfg())
    (o,) = rep.orders
    got = [o.outcomes[p].filled_qty for p in ("conservative", "balanced", "optimistic")]
    assert got == [0, D("0.3"), D("0.9")]


# --- coverage and horizons ---------------------------------------------------------------------------------------


def test_coverage_filter_uses_data_quality_only():
    res = Result()
    for i in range(5):
        res.place(res.request(rq(i)))
    res.gap(100 * S + H, "DESYNC")  # at T1 + H: inside the closed window
    res.gap(200 * S - 1, "RECV_GAP")  # just before T2: outside
    res.gap(300 * S + 3 * S, "RESNAPSHOT")
    res.gap(400 * S + H + 1, "DESYNC")  # just after T4's window
    res.quality.append(QualityEvent(5, SYM, "TS_REGRESSION", None, None, {}, None))  # not a discontinuity
    rep = analyze(res.result(), cfg())
    assert [(r.status, r.reason) for r in rep.requests] == [
        (ELIGIBLE, None), (COVERAGE, "DESYNC"), (ELIGIBLE, None), (COVERAGE, "RESNAPSHOT"), (ELIGIBLE, None)]
    assert rep.request_counts["coverage"] == {"DESYNC": 1, "RESNAPSHOT": 1}
    assert rep.lifecycle["join_best"].eligible == 3


def test_end_of_data_inside_the_window_and_latency():
    res = Result()
    res.place(res.request(rq(1)))
    res.place(res.request(rq(2)))
    res.quality = [QualityEvent(200 * S + H, SYM, "END_OF_DATA", None, None, {}, None)]
    rep = analyze(res.result(), cfg())
    assert [r.status for r in rep.requests] == [ELIGIBLE, ELIGIBLE]  # the item at exactly T + H fired the expiry
    rep = analyze(res.result(), cfg(latency_ns=1))  # latency extends the window by 1 ns
    assert [(r.status, r.reason) for r in rep.requests] == [(ELIGIBLE, None), (COVERAGE, "END_OF_DATA")]


def test_shorter_analysis_horizon():
    res = Result()
    res.place(res.request(rq(0)), fills=[(6 * S, "1.0", LEVEL, MOD)])
    res.place(res.request(rq(1)), end=EndReason.OUT_OF_DEPTH, end_ts=7 * S)  # censored after H' = 5 s
    res.place(res.request(rq(2)), end=EndReason.OUT_OF_DEPTH, end_ts=2 * S)
    rep = analyze(res.result(), cfg(), horizon_ns=5 * S)
    lc, s = rep.lifecycle["join_best"], rep.summary[("join_best", "balanced")]
    assert (lc.censored, lc.censored_after_horizon, lc.expired) == (1, 1, 1)
    assert (s.filled_any, s.unfilled_observed, s.censored_unfilled) == (0, 2, 1)  # the 6 s fill is past H'
    assert analyze(res.result(), cfg()).summary[("join_best", "balanced")].filled_any == 1
    with pytest.raises(ValueError):
        analyze(res.result(), cfg(), horizon_ns=H + 1)


# --- strategy comparison -------------------------------------------------------------------------------------


def test_unequal_placement_frequencies_and_common_support():
    res = Result()
    for i in range(4):
        r = res.request(rq(i))
        res.place(r, "join_best", fills=[(S, "1", LEVEL, MOD)] if i < 2 else ())
        if i == 0:
            res.place(r, "improve_quote", price="100.1", fills=[(S, "1", INTERCEPT, MOD)], ahead=())
        else:
            res.skip(r, "improve_quote", SkipReason.SPREAD_TOO_NARROW)
    rep = analyze(res.result(), cfg(), strategies=["join_best", "improve_quote"])
    assert rep.request_counts["common_support"] == 1
    jb, iq = rep.summary[("join_best", "balanced")], rep.summary[("improve_quote", "balanced")]
    assert rate(iq.fill_given_placed) == (1, 1) and rate(jb.fill_given_placed) == (2, 4)
    assert rate(iq.fill_per_opportunity) == (1, 4) and rate(jb.fill_per_opportunity) == (2, 4)  # (a): same base
    assert rate(iq.fill_common_support) == (1, 1) and rate(jb.fill_common_support) == (1, 1)  # (b)
    rows = {row["strategy"]: row for row in comparison_rows(rep, "balanced")}
    assert rows["improve_quote"]["placed"] == 1 and rows["improve_quote"]["eligible"] == 4
    assert rate(rows["improve_quote"]["placement_rate"]) == (1, 4)
    text = format_report(rep)
    assert "improve_quote" in text and "(1/4)" in text and "SPREAD_TOO_NARROW" in text


def test_queue_metrics_and_buckets():
    res = Result()
    res.place(res.request(rq(0)), fills=[(S, "1", LEVEL, MOD)], ahead=())  # 0 ahead, filled
    res.place(res.request(rq(1)), ahead=(("A", "1"), ("B", "1"), ("C", "2")), touch="4")  # 3 ahead
    res.place(res.request(rq(2)), fills=[(S, "1", LEVEL, MOD)], ahead=(("A", "0.5"),), rank=2, touch="2")
    res.place(res.request(rq(3)), end=EndReason.OUT_OF_DEPTH, end_ts=S, ahead=(("A", "1"),))  # censored
    rep = analyze(res.result(), cfg())
    q, s = rep.queue["join_best"], rep.summary[("join_best", "balanced")]
    assert q.ahead_qty == Dist.of([0, 4, D("0.5"), 1]) and q.ahead_orders == Dist.of([0, 3, 1, 1])
    assert q.level_rank == {1: 3, 2: 1} and q.ahead_to_touch == Dist.of([0, 1, D("0.25"), D(1) / 3])
    assert {k: rate(v) for k, v in s.fill_by_ahead_orders.items()} == {"0": (1, 1), "1-2": (1, 1), "3-5": (0, 1)}
    assert {k: rate(v) for k, v in s.fill_by_level_rank.items()} == {"1": (1, 2), "2": (1, 1)}
    assert q.edge_bps == Dist.of([D("0.1") / D("100.1") * 10_000] * 4)  # buy 100.0 vs decision mid 100.1


# --- markouts ---------------------------------------------------------------------------------------------------


def test_markout_signs_and_missing_future_prices():
    res = Result()
    b, s_ = res.request(rq(1, BUY)), res.request(rq(1, SELL))
    res.place(b, fills=[(0, "1", LEVEL, MOD)])  # buy 100.0 at T
    res.place(s_, fills=[(0, "1", LEVEL, MOD)])  # sell 100.2 at T
    T = b.ts_ns
    m = mids((T + S, "100.1"), (T + 5 * S, "99.9"), (T + 10 * S, "DATA_GAP"))  # nothing at T + 100 ms
    rep = analyze(res.result(), cfg(), m, markout_horizons_ns=(100 * MS, S, 5 * S, 10 * S))
    buy, sell = (next(f for f in rep.fills if f.side is side and f.policy == "balanced") for side in (BUY, SELL))
    assert buy.markout_bps[S] == D(10)  # bought at 100.0, mid rose to 100.1: favorable
    assert sell.markout_bps[S] == (D("100.2") - D("100.1")) / D("100.2") * 10_000  # sold above the later mid
    assert buy.markout_bps[5 * S] == D(-10)  # mid fell below our buy: adverse
    assert sell.markout_bps[5 * S] > 0
    assert buy.markout_bps[S] == signed_markout_bps(BUY, D("100.0"), D("100.1"))
    assert (buy.markout_bps[100 * MS], buy.markout_missing[100 * MS]) == (None, "NOT_SAMPLED")
    assert (buy.markout_bps[10 * S], buy.markout_missing[10 * S]) == (None, "DATA_GAP")
    sm = rep.summary[("join_best", "balanced")].markouts
    assert (sm[S].n_fills, sm[S].n_valid, rate(sm[S].frac_negative)) == (2, 2, (0, 2))
    assert (sm[10 * S].n_valid, sm[10 * S].missing, sm[10 * S].mean) == (0, {"DATA_GAP": 2}, None)
    assert rate(sm[10 * S].frac_negative) == (0, 0)  # missing values are never zero markouts
    assert rate(sm[5 * S].frac_negative) == (1, 2)


def test_markout_mean_median_weighting_and_adverse_selection():
    res = Result()
    r0, r1, r2 = (res.request(rq(i)) for i in range(3))
    res.place(r0, qty="1", fills=[(0, "1", LEVEL, MOD)])
    res.place(r1, qty="3", fills=[(0, "3", LEVEL, MOD)])
    res.place(r2)  # unfilled, still part of the drift baseline
    m = mids((r0.ts_ns + S, "100.2"), (r1.ts_ns + S, "99.9"), (r2.ts_ns + S, "100.3"))
    rep = analyze(res.result(), cfg(), m, markout_horizons_ns=(S,))
    mk = rep.summary[("join_best", "balanced")].markouts[S]
    assert (mk.n_valid, mk.mean, mk.median) == (2, D(5), D(5))  # +20 and -10 bps
    assert mk.qty_weighted_mean == (D(20) * 1 + D(-10) * 3) / 4
    # baseline: buy drift from the decision mid 100.1 over all three requests: (+1, -2, +2) / 100.1 * 1e4 / 3
    base = (D("0.1") - D("0.2") + D("0.2")) / D("100.1") * 10_000 / 3
    assert mk.drift_baseline == base and mk.adverse_selection == -(D(5) - base) and mk.n_adverse == 2


# --- integrity ----------------------------------------------------------------------------------------------------


def test_reaches_without_fill_recording_are_rejected():
    res = Result()
    res.place(res.request(rq(0)), fills=[(S, "1", LEVEL, MOD)])
    res.orders[0].fills.clear()  # as if the replay ran without FillRecorder
    with pytest.raises(ValueError, match="FillRecorder"):
        analyze(res.result(), cfg())


def test_missing_strategy_record_is_rejected():
    res = Result()
    r = res.request(rq(0))
    res.place(r, "join_best")
    with pytest.raises(ValueError, match="no record"):
        analyze(res.result(), cfg(), strategies=["join_best", "quote_deeper"])


def test_analysis_is_deterministic_and_does_not_mutate():
    res = Result()
    for i in range(6):
        r = res.request(rq(i, BUY if i % 2 else SELL))
        res.place(r, fills=[(i * S, "0.5", LEVEL, MOD), ((i + 1) * S, "1", ARRIVAL, CROSS)] if i % 3 else ())
    result = res.result()

    def state(r):
        return ([(o.order_id, o.state, o.inserted_ts, o.end_ts, o.end_reason, o.skip_reason, list(o.fills),
                  o.queue_ahead_qty0, dict(o.diag)) for o in r.orders], list(r.reaches), list(r.skips),
                list(r.quality), dict(r.views))

    before = copy.deepcopy(state(result))
    m = mids(*((r.ts_ns + d, "100.1") for r in (rq(i, s) for i in range(6) for s in (BUY, SELL))
               for d in range(0, 20 * S, S // 2)))
    a, b = analyze(result, cfg(), m), analyze(result, cfg(), m)
    assert a == b and a.summary == b.summary and a.fills == b.fills
    assert state(result) == before


# --- engine integration ------------------------------------------------------------------------------------------


def test_sample_mids_follow_the_engine_gate(tmp_path):
    tape = Tape()
    tape.update(0)  # before any snapshot
    tape.snapshot(10)  # bid 100.0, ask 100.2
    tape.update(20, asks=[order("A1", "100.2", "3.0", "delete")])  # one-sided from t=20
    tape.update(30, asks=[order("A2", "100.4", "1.0", "add")])
    tape.update(40)
    files = tape.write(tmp_path)
    got = sample_mids(engine([]).cfg, [(SYM, at(t)) for t in (5, 10, 15, 20, 25, 35, 40, 99)], files=files)
    assert [(s.mid, s.missing) for s in (got[(SYM, at(t))] for t in (5, 10, 15, 20, 25, 35, 40, 99))] == [
        (None, "UNSYNCED"), (None, "UNSYNCED"),  # the snapshot at 10 is not yet visible at 10 (strict-before)
        (D("100.1"), None), (D("100.1"), None),
        (None, "ONE_SIDED"), (D("100.2"), None), (D("100.2"), None),
        (None, "NO_DATA")]  # after the last item: no mid, never forward-filled


def test_evaluate_end_to_end_on_a_tape(tmp_path):
    tape = Tape()
    tape.snapshot(0, bids=BIDS + [order("C1", "99.9", "1.0")])
    tape.update(100, bids=[order("B1", "100.0", "1.0", "delete"), order("B2", "100.0", "1.5", "modify")])
    tape.update(200, bids=[order("B2", "100.0", "1.0", "modify")])  # touch reduced below improve_quote's 100.1
    tape.update(300, asks=[order("A2", "100.1", "0.2", "add")])  # crosses improve_quote's buy
    for t in range(400, 3_000, 100):
        if t == 2_300:  # B2 leaves: join_best's 1.5 s buy at 100.0 is now first in line
            tape.update(t, bids=[order("B2", "100.0", "1.0", "delete")])
        elif t == 2_400:  # an ask arrives on that buy (CROSS_ARRIVAL), then leaves
            tape.update(t, asks=[order("A3", "100.0", "0.1", "add")])
        elif t == 2_500:
            tape.update(t, asks=[order("A3", "100.0", "0.1", "delete")])
        else:
            tape.update(t)
    files = tape.write(tmp_path)
    schedule = [req(5), req(5, SELL), req(1_500), req(2_500)]
    strategies = [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()]
    from toxicflow.execsim.config import SimConfig
    c = SimConfig(symbols=(SYM,), order_qty={SYM: "0.5"}, horizon_ns=1 * S, audit_invariants=True)
    result, rep = evaluate(c, strategies, schedule, files, markout_horizons_ns=(100 * MS, 1 * S, 2 * S))
    # 1.5 s: the window ends at 2.5 s, inside the data (last item 2.9 s); 2.5 s: it would end at 3.5 s.
    # improve_quote skips at 1.5 s (spread is one tick after A2), so common support is the two requests at 5 ms
    assert rep.request_counts == {"total": 4, "eligible": 3, "common_support": 2, "data_gate": {},
                                  "coverage": {"END_OF_DATA": 1}}
    assert rep.lifecycle["improve_quote"].strategy_skip_reasons == {"SPREAD_TOO_NARROW": 1}
    for name in ("join_best", "improve_quote", "quote_deeper"):
        lc = rep.lifecycle[name]
        assert lc.eligible == lc.strategy_skips + lc.insertion_skips + lc.placed
        assert lc.placed == lc.expired + lc.ended_all_filled + lc.censored + lc.censored_after_horizon
    iq = rep.summary[("improve_quote", "balanced")]
    assert (iq.filled_any, iq.fills_by_mechanism) == (1, {"intercept": 1})
    iq_fill = next(f for f in rep.fills if f.strategy == "improve_quote" and f.policy == "balanced")
    assert (iq_fill.ts_ns, iq_fill.qty, iq_fill.ahead_qty) == (at(100), D("0.5"), 0)
    # mid after the fill: bid 100.0, ask 100.2 -> 100.1 = our buy price: zero markout, a valid observation
    assert iq_fill.markout_bps[100 * MS] == 0
    # the last fill + 2 s runs past the data: NO_DATA, not zero
    (late,) = [f for f in rep.fills if f.ts_ns == at(2_400) and f.policy == "optimistic"]
    assert (late.strategy, late.mechanism, late.qty) == ("join_best", ARRIVAL, D("0.1"))
    # +100 ms: the ask still sits on the best bid (99.9 < 100.0 = ask): mid 99.95, a valid observation;
    # +1 s and +2 s run past the last item (2.9 s): missing, never zero
    assert late.markout_missing == {100 * MS: None, S: "NO_DATA", 2 * S: "NO_DATA"}
    assert late.markout_bps[100 * MS] == (D("99.95") - D("100.0")) / D("100.0") * 10_000 < 0
    mk = rep.summary[("join_best", "optimistic")].markouts
    assert mk[2 * S].missing == {"NO_DATA": 1} and mk[2 * S].n_valid == mk[2 * S].n_fills - 1
    assert required_mid_times(result, c, markout_horizons_ns=(S,)) >= {(SYM, at(5) + S)}
    assert "join_best" in format_report(rep)
