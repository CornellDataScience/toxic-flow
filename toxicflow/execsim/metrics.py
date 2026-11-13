"""Strategy evaluation metrics from a ReplayResult (design §11.4, §11.5).

Pipeline (evaluate() does all of it):

    result = HistoricalReplayEngine(cfg, strategies, schedule, on_reach=FillRecorder()).run(files)
    mids = sample_mids(cfg, required_mid_times(result, cfg), files=files)
    report = analyze(result, cfg, mids)
    print(format_report(report))

Everything is derived from the ReplayResult; nothing in it is modified.

Request eligibility (shared by all strategies, so comparisons stay paired, §11.4)
    DATA_GATE   the engine's data gate refused the decision (UNSYNCED, WARMUP, DATA_GAP, NO_DATA):
                no BookView exists for it
    COVERAGE    the coverage filter: a discontinuity (DISCONTINUITY_KINDS) for the request's symbol
                at a time in [T, T + latency + H], or the data ending before T + latency + H.
                Depends on data quality only, never on fills.
    ELIGIBLE    everything else: the decision opportunities every denominator starts from

Per (eligible request, strategy) the stages are disjoint:
    eligible = strategy_skips + orders_created
    orders_created = insertion_skips + placed            (insertion skips: CROSSED_AT_INSERT, OUT_OF_DEPTH, ...)
    placed = expired + ended_all_filled + censored + censored_after_horizon

A placed order is *censored* when its end reason is a censor (CENSORED_*, OUT_OF_DEPTH) and it ended
before inserted + H. Per policy, a placed order is then exactly one of
    filled_any         filled > 0 at H (also when censored later: fills before a censor are observed)
    unfilled_observed  filled == 0 and observed for the whole horizon
    censored_unfilled  filled == 0 and censored: outcome unknown, never counted as a non-fill

Rates are Rate(num, den); value is None when den == 0 (never a silent 0).

Evidence modes (analyze(evidence=...)); one replay supports both when it ran with
FillRecorder(attribution):
    l3_only   policies apply the L3-only thresholds to each candidate's L3 label (the Stage 4
              meaning; the default)
    trades    policies apply the with-trades thresholds to each candidate's trade slices
              (trades.py): conservative = TRADE_CONFIRMED, balanced >= TRADE_PLAUSIBLE,
              optimistic >= L3_CROSS_ARRIVAL; TRADE_ABSENT (an unmatched reduction while the
              trade feed is healthy) never counts

Crossing rules (analyze(cross_rule=...), execution.CROSS_RULES); one replay supports all three.
An order is crossed when an opposite-side order comes to rest at or through its price before
inserted + H. Under `stop` (and under `fill_at_cross` for the policies that do not accept a
crossing arrival) a crossed order that was not complete by then is `resolved_by_cross`: per
policy it is treated exactly like a censored order (outcome unknown, never a non-fill, removed
from the point-estimate denominators, censored at the crossing time in Kaplan-Meier) but is
counted separately (crossed_unfilled), because it is a property of the market path, not of data
quality. Crossing is related to the outcome (it happens when the price moves through us), so
read fill_given_placed together with its lower and upper bounds. The library default is
`ignore` (the earlier behaviour); reports and the runner default to REPORTING_CROSS_RULE.
"""

import bisect
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from .config import SimConfig
from .execution import CROSS_RULES, POLICY_NAMES, FillRecorder, PolicyOutcome, settle
from .markouts import (DEFAULT_MARKOUT_HORIZONS_NS, MidSample, drift_bps, edge_bps, sample_mids, signed_markout_bps,
                       view_mid)
from .models import EndReason, Evidence, EvidencePart, Mechanism, PlacementRequest, Side, SkipReason
from .orders import OrderState
from .replay import HistoricalReplayEngine
from .results import DISCONTINUITY_KINDS, ReplayResult, SkipRecord
from .strategies import OrderPlacementStrategy
from .timeutil import MS
from .trades import DEFAULT_TOLERANCE, MatchTolerance, TradeIndex, attribute_trades

ELIGIBLE, DATA_GATE, COVERAGE = "ELIGIBLE", "DATA_GATE", "COVERAGE"
EVIDENCE_MODES = ("l3_only", "trades")
SKIPPED, INSERT_SKIPPED, PLACED = "SKIPPED", "INSERT_SKIPPED", "PLACED"
AHEAD_ORDER_BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, None, "11+"))
LEVEL_RANK_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, None, "3+"))


# --- small value types ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Rate:
    num: int
    den: int

    @property
    def value(self) -> float | None:
        """num / den, or None when den == 0 (undefined, not zero)."""
        return None if self.den == 0 else self.num / self.den

    def __str__(self) -> str:
        v = self.value
        return f"{'n/a' if v is None else f'{v:.3f}'} ({self.num}/{self.den})"


@dataclass(frozen=True, slots=True)
class Dist:
    """Sample summary. Statistics are None for an empty sample."""

    n: int
    mean: Decimal | None
    median: Decimal | None
    min: Decimal | None
    max: Decimal | None

    @classmethod
    def of(cls, values: Iterable) -> "Dist":
        xs = sorted(Decimal(v) for v in values)
        if not xs:
            return cls(0, None, None, None, None)
        n, mid = len(xs), len(xs) // 2
        median = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2
        return cls(n, sum(xs, Decimal(0)) / n, median, xs[0], xs[-1])


@dataclass(frozen=True, slots=True)
class KaplanMeier:
    """Kaplan-Meier estimate of time to first fill (§11.4): a fill is the event; unfilled at the
    horizon is censored at H, a censored order at its end. Times in ns since insertion."""

    n: int
    n_events: int
    n_censored: int
    median_ns: int | None  # first time the survival drops to 1/2 or below; None if it never does
    fill_prob_at_horizon: float | None  # 1 - S(H)
    curve: tuple[tuple[int, float], ...]  # (time, survival) after each event time


def kaplan_meier(obs: Iterable[tuple[int, bool]]) -> KaplanMeier:
    """obs: (duration_ns, event). At tied times events are counted before censorings."""
    obs = sorted(obs)
    n = len(obs)
    if n == 0:
        return KaplanMeier(0, 0, 0, None, None, ())
    at_risk, surv, curve, median, i = n, Fraction(1), [], None, 0
    while i < n:
        t = obs[i][0]
        j = i
        while j < n and obs[j][0] == t:
            j += 1
        d = sum(1 for _, e in obs[i:j] if e)
        if d:
            surv *= 1 - Fraction(d, at_risk)
            curve.append((t, float(surv)))
            if median is None and surv <= Fraction(1, 2):
                median = t
        at_risk -= j - i
        i = j
    events = sum(1 for _, e in obs if e)
    return KaplanMeier(n, events, n - events, median, float(1 - surv), tuple(curve))


# --- raw observations -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RequestObs:
    request: PlacementRequest
    status: str  # ELIGIBLE | DATA_GATE | COVERAGE
    reason: str | None  # gate SkipReason or the first coverage break (e.g. DESYNC, END_OF_DATA)
    decision_mid: Decimal | None  # from the shared BookView; None without a view or with a one-sided book


@dataclass(frozen=True, slots=True)
class OrderObs:
    """One (request, strategy) pair."""

    request: PlacementRequest
    strategy: str
    eligible: bool
    stage: str  # SKIPPED | INSERT_SKIPPED | PLACED
    skip_reason: SkipReason | None
    order_id: str | None = None
    price: Decimal | None = None
    qty: Decimal | None = None
    inserted_ts: int | None = None
    end_ts: int | None = None
    end_reason: EndReason | None = None
    censored: bool = False  # censored before inserted + H
    queue_ahead_qty0: Decimal | None = None  # observed at insertion only
    queue_ahead_orders0: int | None = None
    level_rank0: int | None = None
    touch_qty0: Decimal | None = None
    edge_bps: Decimal | None = None  # vs the decision mid
    outcomes: dict[str, PolicyOutcome] | None = None  # per policy, placed orders only
    cross_ts: int | None = None  # first crossing before inserted + H (policy and rule independent)
    cross_ahead_qty: Decimal | None = None  # our queue ahead at the crossing
    inconsistent_crossing: bool | None = None  # crossed with no reachable reduction up to then (§6.8)
    legacy_outcomes: dict[str, PolicyOutcome] | None = None  # crossed orders: settlement under `ignore`


@dataclass(frozen=True, slots=True)
class FillObs:
    """One settled fill (the part of a candidate a policy counts) of an eligible placed order."""

    policy: str
    strategy: str
    order_id: str
    symbol: str
    side: Side
    ts_ns: int
    since_insert_ns: int
    price: Decimal
    qty: Decimal  # settled (clipped) quantity
    candidate_qty: Decimal  # historical volume of the reduction, unclipped
    mechanism: Mechanism
    evidence: Evidence
    event_ref: tuple
    ahead_qty: Decimal | None  # tracker's queue ahead when the reduction reached us (0 for LEVEL)
    markout_bps: dict[int, Decimal | None]  # horizon_ns -> signed markout; None when the future mid is missing
    markout_missing: dict[int, str | None]  # horizon_ns -> why it is missing
    qty_by_evidence: dict[str, Decimal] = None  # settled qty split by the evidence that justified it
    after_cross: bool = False  # from an item later than the order's first crossing


# --- summaries -------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Lifecycle:
    """Policy-independent counts for one strategy over eligible requests."""

    eligible: int
    strategy_skips: int
    strategy_skip_reasons: dict[str, int]
    orders_created: int
    insertion_skips: int
    insertion_skip_reasons: dict[str, int]
    placed: int
    expired: int  # HORIZON
    ended_all_filled: int  # ALL_POLICIES_FILLED
    censored: int  # censor end reason before inserted + H
    censored_reasons: dict[str, int]
    censored_after_horizon: int  # censor end reason at or after inserted + H (only when H < run horizon)
    placement_rate: Rate  # placed / eligible
    censored_rate: Rate  # censored / placed
    crossed: int = 0  # placed orders crossed before inserted + H (policy and rule independent)
    inconsistent_crossings: int = 0  # of those, with no reachable reduction up to the crossing


@dataclass(frozen=True, slots=True)
class QueueStats:
    """Queue position at insertion, the only time it is fully observed (placed eligible orders)."""

    ahead_qty: Dist
    ahead_orders: Dist
    ahead_to_touch: Dist  # queue_ahead_qty0 / touch_qty0 where touch_qty0 > 0
    level_rank: dict[int, int]
    edge_bps: Dist


@dataclass(frozen=True, slots=True)
class MarkoutSummary:
    horizon_ns: int
    n_fills: int  # settled fills
    n_valid: int  # with a future mid
    missing: dict[str, int]  # reason -> count; never included as zero markouts
    mean: Decimal | None
    median: Decimal | None
    qty_weighted_mean: Decimal | None
    frac_negative: Rate  # markout < 0 among valid
    drift_baseline: Decimal | None  # mean no-fill signed mid drift for the same (symbol, side) mix
    adverse_selection: Decimal | None  # -(mean markout - drift baseline) over fills with both (§11.5)
    n_adverse: int


@dataclass(frozen=True, slots=True)
class FillSummary:
    """One strategy under one policy."""

    strategy: str
    policy: str
    filled_any: int
    filled_full: int
    partial: int  # 0 < filled < qty at H (includes censored partials)
    unfilled_observed: int
    censored_unfilled: int
    full_undetermined: int  # censored with filled < qty: whether it would have completed is unknown
    fill_given_placed: Rate  # filled_any / (placed - censored_unfilled)
    fill_given_placed_lower: Rate  # filled_any / placed (censored as non-fills)
    fill_given_placed_upper: Rate  # (filled_any + censored_unfilled) / placed (censored as fills)
    fill_per_opportunity: Rate  # filled_any / (eligible - censored_unfilled), §11.4 (a)
    fill_common_support: Rate  # same, on requests where every strategy placed, §11.4 (b)
    full_fill_given_placed: Rate  # filled_full / (placed - full_undetermined)
    full_fill_per_opportunity: Rate  # filled_full / (eligible - full_undetermined)
    n_fills: int
    fills_by_mechanism: dict[str, int]
    time_to_first_fill_ns: Dist  # naive: filled orders only
    time_to_complete_fill_ns: Dist  # naive: fully filled orders only
    km_first_fill: KaplanMeier
    fill_by_ahead_orders: dict[str, Rate]  # bucket of queue_ahead_orders0 -> fill_given_placed
    fill_by_level_rank: dict[str, Rate]
    ahead_qty_at_fill: Dist  # observed at the reach event only
    markouts: dict[int, MarkoutSummary]
    fill_qty_by_evidence: dict[str, Decimal] = None  # settled volume by evidence class
    fills_by_evidence: dict[str, int] = None  # settled fills by their strongest evidence class
    # crossing (see the module docstring). censored_unfilled excludes crossed_unfilled; both are
    # outside the point-estimate denominators and inside the lower / upper bounds.
    resolved_by_cross: int = 0  # orders this policy stopped observing at a crossing, still incomplete
    crossed_unfilled: int = 0  # of those, with nothing filled before the crossing
    fills_after_cross: int = 0  # settled fills from items after the order's crossing (this rule)
    legacy_filled_any: int = 0  # filled_any as it would be under `ignore`
    legacy_fills_after_cross: int = 0  # settled fills after the crossing under `ignore`
    legacy_first_fill_after_cross: int = 0  # orders whose first fill under `ignore` is after it


@dataclass(frozen=True, slots=True)
class EvidenceStats:
    """Every candidate (policy independent) of one strategy's eligible placed orders before
    inserted + H, by evidence class: L3 labels in l3_only mode, trade slices in trades mode.
    In trades mode L3_DELETE_AMBIGUOUS / L3_MODIFY_DOWN slices are the ambiguous ones (no
    healthy trade feed there) and TRADE_ABSENT slices are reductions with no execution evidence."""

    n_candidates: int
    candidates_by_evidence: dict[str, int]  # candidates holding any slice of that class
    qty_by_evidence: dict[str, Decimal]  # unclipped historical volume


@dataclass
class EvaluationReport:
    horizon_ns: int
    latency_ns: int
    strategies: tuple[str, ...]
    policies: tuple[str, ...]
    markout_horizons_ns: tuple[int, ...]
    requests: list[RequestObs]
    orders: list[OrderObs]
    fills: list[FillObs]
    request_counts: dict  # total, eligible, common_support, data_gate {reason: n}, coverage {kind: n}
    lifecycle: dict[str, Lifecycle]
    queue: dict[str, QueueStats]
    summary: dict[tuple[str, str], FillSummary]  # (strategy, policy)
    evidence_mode: str = "l3_only"
    evidence: dict[str, EvidenceStats] = field(default_factory=dict)
    attribution: dict | None = None  # TradeAttribution.stats (recording-wide), when supplied
    cross_rule: str = "ignore"


# --- analysis ----------------------------------------------------------------------------------------------------


def evaluate(cfg: SimConfig, strategies: Sequence[OrderPlacementStrategy], schedule: Iterable[PlacementRequest],
             files: Sequence[Path], horizon_ns: int | None = None,
             markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
             policies: Sequence[str] = POLICY_NAMES, trade_files: Sequence[Path] | None = None,
             evidence: str = "l3_only", tolerance: MatchTolerance = DEFAULT_TOLERANCE,
             cross_rule: str = "ignore") -> tuple[ReplayResult, EvaluationReport]:
    """Replay with fill recording, sample the mids the markouts need (second pass), and analyze.

    evidence='trades' needs `trade_files` (public-stream recordings covering the same hours).
    """
    if evidence == "trades" and not trade_files:
        raise ValueError("evidence='trades' needs trade_files")
    result, reports = evaluate_modes(cfg, strategies, schedule, files, horizon_ns, markout_horizons_ns, policies,
                                     trade_files, tolerance, modes=(evidence,), cross_rule=cross_rule)
    return result, reports[evidence]


def evaluate_modes(cfg: SimConfig, strategies: Sequence[OrderPlacementStrategy],
                   schedule: Iterable[PlacementRequest], files: Sequence[Path], horizon_ns: int | None = None,
                   markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
                   policies: Sequence[str] = POLICY_NAMES, trade_files: Sequence[Path] | None = None,
                   tolerance: MatchTolerance = DEFAULT_TOLERANCE, modes: Sequence[str] | None = None,
                   cross_rule: str = "ignore") -> tuple[ReplayResult, dict[str, EvaluationReport]]:
    """One replay, one mid-sampling pass, one report per evidence mode (default: 'l3_only', plus
    'trades' when `trade_files` are given), so the two modes are compared on identical orders."""
    attribution = None
    if trade_files:
        index = TradeIndex.from_files(trade_files, cfg.symbols, gap_ns=cfg.gap_ns, tolerance=tolerance)
        attribution = attribute_trades(cfg, index, files=files)
    eng = HistoricalReplayEngine(cfg, strategies, schedule, on_reach=FillRecorder(attribution))
    result = eng.run(files)
    if modes is None:
        modes = EVIDENCE_MODES if attribution is not None else ("l3_only",)
    times = required_mid_times(result, cfg, horizon_ns, markout_horizons_ns, policies, evidence=modes,
                               cross_rule=cross_rule)
    mids = sample_mids(cfg, times, files=files)
    reports = {m: analyze(result, cfg, mids, horizon_ns, markout_horizons_ns, policies,
                          strategies=[s.name for s in strategies], evidence=m, attribution=attribution,
                          cross_rule=cross_rule)
               for m in modes}
    return result, reports


def required_mid_times(result: ReplayResult, cfg: SimConfig, horizon_ns: int | None = None,
                       markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
                       policies: Sequence[str] = POLICY_NAMES,
                       evidence: str | Sequence[str] = "l3_only",
                       cross_rule: str | Sequence[str] = "ignore") -> set[tuple[str, int]]:
    """(symbol, ts) pairs analyze() needs mids for: every settled fill + D (in each evidence mode
    and under each crossing rule given), and every eligible request T + D."""
    times: set[tuple[str, int]] = set()
    for mode in (evidence,) if isinstance(evidence, str) else evidence:
        for rule in (cross_rule,) if isinstance(cross_rule, str) else cross_rule:
            report = analyze(result, cfg, None, horizon_ns, markout_horizons_ns, policies, evidence=mode,
                             cross_rule=rule)
            times |= {(f.symbol, f.ts_ns + d) for f in report.fills for d in report.markout_horizons_ns}
            times |= {(r.request.symbol, r.request.ts_ns + d) for r in report.requests if r.status == ELIGIBLE
                      for d in report.markout_horizons_ns}
    return times


def analyze(result: ReplayResult, cfg: SimConfig, mids: dict[tuple[str, int], MidSample] | None = None,
            horizon_ns: int | None = None, markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
            policies: Sequence[str] = POLICY_NAMES, strategies: Sequence[str] | None = None,
            evidence: str = "l3_only", attribution=None, cross_rule: str = "ignore") -> EvaluationReport:
    """Metrics for every strategy in `result` at analysis horizon H = `horizon_ns` (default: the run's).

    `result` must come from a replay with FillRecorder as `on_reach` (with a TradeAttribution for
    evidence='trades'); without `mids`, every markout is missing ('NOT_SAMPLED'). `attribution`
    only adds its recording-wide accounting to the report.
    """
    if evidence not in EVIDENCE_MODES:
        raise ValueError(f"evidence must be one of {EVIDENCE_MODES}, got {evidence!r}")
    if cross_rule not in CROSS_RULES:
        raise ValueError(f"cross_rule must be one of {CROSS_RULES}, got {cross_rule!r}")
    H = cfg.horizon_ns if horizon_ns is None else horizon_ns
    if not 0 < H <= cfg.horizon_ns:
        raise ValueError(f"analysis horizon must be in (0, {cfg.horizon_ns}], got {horizon_ns}")
    horizons = tuple(markout_horizons_ns)
    if any(d <= 0 for d in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError(f"markout horizons must be positive and unique, got {horizons}")
    _check_fill_recording(result, with_trades=evidence == "trades")

    records: dict[tuple[str, str], object] = {}
    reqs: dict[str, PlacementRequest] = {}
    gate_reason: dict[str, str] = {}
    for rec in [*result.orders, *result.skips]:
        rid = rec.request.request_id
        key = (rid, rec.strategy)
        if key in records:
            raise ValueError(f"two records for request {rid}, strategy {rec.strategy}")
        records[key] = rec
        reqs[rid] = rec.request
        if rid not in result.views and isinstance(rec, SkipRecord):
            gate_reason[rid] = rec.reason.value
    names = tuple(strategies) if strategies is not None else tuple(sorted({k[1] for k in records}))
    missing = [k for k in ((rid, s) for rid in reqs for s in names) if k not in records]
    if missing:
        raise ValueError(f"no record for {len(missing)} (request, strategy) pairs, e.g. {missing[0]}")

    breaks = _coverage_breaks(result)
    requests: list[RequestObs] = []
    for req in sorted(reqs.values(), key=lambda r: (r.ts_ns, r.symbol, r.side.value, r.request_id)):
        view = result.views.get(req.request_id)
        mid = view_mid(view)[0] if view is not None else None
        if view is None:
            requests.append(RequestObs(req, DATA_GATE, gate_reason.get(req.request_id, "NO_VIEW"), None))
            continue
        brk = _coverage_break(breaks.get(req.symbol), req.ts_ns, req.ts_ns + cfg.latency_ns + H)
        requests.append(RequestObs(req, COVERAGE if brk else ELIGIBLE, brk, mid))

    reach_ahead = {(r.order_id, r.reach.ref): r.ahead_qty for r in result.reaches}
    orders: list[OrderObs] = []
    fills: list[FillObs] = []
    for r in requests:
        for name in names:
            obs = _order_obs(records[(r.request.request_id, name)], r, H, policies, evidence == "trades",
                             cross_rule)
            orders.append(obs)
            if obs.eligible and obs.stage == PLACED:
                fills.extend(_fill_obs(obs, policies, reach_ahead, mids, horizons))

    baseline = _drift_baseline(requests, mids, horizons)
    report = EvaluationReport(H, cfg.latency_ns, names, tuple(policies), horizons, requests, orders, fills, {}, {},
                              {}, {}, evidence, {}, attribution.stats if attribution is not None else None,
                              cross_rule)
    _summarize(report, baseline)
    report.evidence = _evidence_stats(report, records)
    return report


def _check_fill_recording(result: ReplayResult, with_trades: bool = False) -> None:
    reaches = Counter(r.order_id for r in result.reaches)
    for o in result.orders:
        if o.state is not OrderState.ENDED:
            raise ValueError(f"order {o.order_id} is {o.state.value}: call finish() before analyzing")
        if len(o.fills) != reaches.get(o.order_id, 0):
            raise ValueError(f"order {o.order_id} has {reaches.get(o.order_id, 0)} reaches but {len(o.fills)} "
                             "fill candidates: replay with on_reach=FillRecorder()")
        refs = [f.event_ref for f in o.fills]
        if len(set(refs)) != len(refs):
            raise ValueError(f"order {o.order_id} has two candidates from the same event")
        for f in o.fills:
            if f.attribution is None:
                if with_trades:
                    raise ValueError(f"order {o.order_id} has candidates without trade attribution: "
                                     "replay with FillRecorder(attribution) for evidence='trades'")
            elif sum((p.qty for p in f.attribution), Decimal(0)) != f.qty:
                raise ValueError(f"order {o.order_id}: trade slices of {f.event_ref} do not sum to its volume")


def _coverage_breaks(result: ReplayResult) -> dict[str, tuple[list[int], list[str], int | None]]:
    """Per symbol: sorted discontinuity times and kinds, and the end-of-data time."""
    raw: dict[str, list[tuple[int, str]]] = defaultdict(list)
    end: dict[str, int] = {}
    for q in result.quality:
        if q.kind in DISCONTINUITY_KINDS:
            raw[q.symbol].append((q.ts_ns, q.kind))
        elif q.kind == "END_OF_DATA":
            end[q.symbol] = q.ts_ns
    out = {}
    for sym in set(raw) | set(end):
        items = sorted(raw.get(sym, []))
        out[sym] = ([t for t, _ in items], [k for _, k in items], end.get(sym))
    return out


def _coverage_break(breaks, start: int, stop: int) -> str | None:
    """The first discontinuity at a time in [start, stop], or END_OF_DATA if the data ends before stop."""
    if breaks is None:
        return None
    times, kinds, end = breaks
    i = bisect.bisect_left(times, start)
    if i < len(times) and times[i] <= stop:
        return kinds[i]
    if end is not None and end < stop:
        return "END_OF_DATA"
    return None


def _order_obs(rec, r: RequestObs, H: int, policies: Sequence[str], with_trades: bool,
               cross_rule: str = "ignore") -> OrderObs:
    eligible = r.status == ELIGIBLE
    if isinstance(rec, SkipRecord):  # SkipRecord: strategy skip (or the data gate for every strategy)
        return OrderObs(r.request, rec.strategy, eligible, SKIPPED, rec.reason)
    o = rec
    if o.inserted_ts is None:
        return OrderObs(r.request, o.strategy, eligible, INSERT_SKIPPED, o.skip_reason, o.order_id, o.price, o.qty)
    censored = o.end_reason.censored and o.end_ts < o.inserted_ts + H
    edge = edge_bps(o.side, o.price, r.decision_mid) if r.decision_mid is not None else None
    until = o.inserted_ts + H
    outcomes = {p: settle(o, p, until_ns=until, with_trades=with_trades, cross_rule=cross_rule) for p in policies}
    cross = o.cross if o.cross is not None and o.cross.ts_ns < min(until, o.expiry_ts) else None
    legacy = None
    if cross is not None:  # what the `ignore` rule would have reported, to measure the rule's effect
        legacy = outcomes if cross_rule == "ignore" else {
            p: settle(o, p, until_ns=until, with_trades=with_trades, cross_rule="ignore") for p in policies}
    return OrderObs(r.request, o.strategy, eligible, PLACED, None, o.order_id, o.price, o.qty, o.inserted_ts,
                    o.end_ts, o.end_reason, censored, o.queue_ahead_qty0, o.queue_ahead_orders0, o.level_rank0,
                    o.touch_qty0, edge, outcomes,
                    cross.ts_ns if cross else None, cross.ahead_qty if cross else None,
                    o.inconsistent_crossing if cross else None, legacy)


def _fill_obs(o: OrderObs, policies, reach_ahead, mids, horizons) -> list[FillObs]:
    out = []
    for p in policies:
        for sf in o.outcomes[p].fills:
            c = sf.candidate
            marks, why = {}, {}
            for d in horizons:
                sample = None if mids is None else mids.get((o.request.symbol, c.ts_ns + d))
                if sample is None:
                    marks[d], why[d] = None, "NOT_SAMPLED"
                elif sample.mid is None:
                    marks[d], why[d] = None, sample.missing
                else:
                    marks[d], why[d] = signed_markout_bps(o.request.side, c.price, sample.mid), None
            out.append(FillObs(p, o.strategy, o.order_id, o.request.symbol, o.request.side, c.ts_ns,
                               c.ts_ns - o.inserted_ts, c.price, sf.qty, c.qty, c.mechanism, c.evidence, c.event_ref,
                               reach_ahead.get((o.order_id, c.event_ref)), marks, why,
                               _sum_by_evidence(sf.by_evidence),
                               o.cross_ts is not None and c.ts_ns > o.cross_ts))
    return out


def _drift_baseline(requests, mids, horizons) -> dict[tuple[str, Side, int], Decimal]:
    """Mean signed mid drift over eligible requests, per (symbol, side, horizon): the no-fill benchmark."""
    acc: dict[tuple[str, Side, int], list[Decimal]] = defaultdict(list)
    if mids is None:
        return {}
    for r in requests:
        if r.status != ELIGIBLE or r.decision_mid is None:
            continue
        for d in horizons:
            s = mids.get((r.request.symbol, r.request.ts_ns + d))
            if s is not None and s.mid is not None:
                acc[(r.request.symbol, r.request.side, d)].append(drift_bps(r.request.side, r.decision_mid, s.mid))
    return {k: sum(v, Decimal(0)) / len(v) for k, v in acc.items()}


def _bucket(value: int | None, buckets) -> str | None:
    if value is None:
        return None
    for lo, hi, label in buckets:
        if value >= lo and (hi is None or value <= hi):
            return label
    return None


def _summarize(report: EvaluationReport, baseline) -> None:
    reqs = report.requests
    eligible_ids = {r.request.request_id for r in reqs if r.status == ELIGIBLE}
    by_strategy: dict[str, list[OrderObs]] = defaultdict(list)
    for o in report.orders:
        if o.eligible:
            by_strategy[o.strategy].append(o)
    placed_by_req: dict[str, int] = Counter(o.request.request_id for o in report.orders
                                            if o.eligible and o.stage == PLACED)
    common = {rid for rid in eligible_ids if placed_by_req.get(rid, 0) == len(report.strategies)}
    report.request_counts = {
        "total": len(reqs),
        "eligible": len(eligible_ids),
        "common_support": len(common),
        "data_gate": dict(sorted(Counter(r.reason for r in reqs if r.status == DATA_GATE).items())),
        "coverage": dict(sorted(Counter(r.reason for r in reqs if r.status == COVERAGE).items())),
    }
    fills_by: dict[tuple[str, str], list[FillObs]] = defaultdict(list)
    for f in report.fills:
        fills_by[(f.strategy, f.policy)].append(f)

    for name in report.strategies:
        obs = by_strategy.get(name, [])
        placed = [o for o in obs if o.stage == PLACED]
        skipped = [o for o in obs if o.stage == SKIPPED]
        ins = [o for o in obs if o.stage == INSERT_SKIPPED]
        censored = [o for o in placed if o.censored]
        report.lifecycle[name] = Lifecycle(
            eligible=len(obs),
            strategy_skips=len(skipped),
            strategy_skip_reasons=_counts(o.skip_reason.value for o in skipped),
            orders_created=len(ins) + len(placed),
            insertion_skips=len(ins),
            insertion_skip_reasons=_counts(o.skip_reason.value for o in ins),
            placed=len(placed),
            expired=sum(o.end_reason is EndReason.HORIZON for o in placed),
            ended_all_filled=sum(o.end_reason is EndReason.ALL_POLICIES_FILLED for o in placed),
            censored=len(censored),
            censored_reasons=_counts(o.end_reason.value for o in censored),
            censored_after_horizon=sum(o.end_reason.censored and not o.censored for o in placed),
            placement_rate=Rate(len(placed), len(obs)),
            censored_rate=Rate(len(censored), len(placed)),
            crossed=sum(o.cross_ts is not None for o in placed),
            inconsistent_crossings=sum(bool(o.inconsistent_crossing) for o in placed),
        )
        report.queue[name] = QueueStats(
            ahead_qty=Dist.of(o.queue_ahead_qty0 for o in placed),
            ahead_orders=Dist.of(o.queue_ahead_orders0 for o in placed),
            ahead_to_touch=Dist.of(o.queue_ahead_qty0 / o.touch_qty0 for o in placed if o.touch_qty0),
            level_rank=dict(sorted(Counter(o.level_rank0 for o in placed).items())),
            edge_bps=Dist.of(o.edge_bps for o in placed if o.edge_bps is not None),
        )
        for p in report.policies:
            report.summary[(name, p)] = _fill_summary(report, name, p, obs, placed, common,
                                                      fills_by.get((name, p), []), baseline)


def _fill_summary(report, name, p, obs, placed, common, fills, baseline) -> FillSummary:
    H = report.horizon_ns

    def filled(o):
        return o.outcomes[p].filled_qty > 0

    def full(o):
        return o.outcomes[p].filled_qty == o.qty

    def resolved(o):  # this policy stopped observing at a crossing, order still incomplete
        return o.outcomes[p].resolved_by_cross

    def cut(o):  # the outcome is not observed to the horizon: data censoring or a crossing
        return o.censored or resolved(o)

    any_ = [o for o in placed if filled(o)]
    full_ = [o for o in placed if full(o)]
    cross_unf = [o for o in placed if resolved(o) and not filled(o)]
    cens_unf = [o for o in placed if o.censored and not resolved(o) and not filled(o)]
    unknown = len(cens_unf) + len(cross_unf)
    full_und = [o for o in placed if cut(o) and not full(o)]
    obs_common = [o for o in obs if o.request.request_id in common]
    common_any = sum(o.stage == PLACED and filled(o) for o in obs_common)
    common_cu = sum(o.stage == PLACED and cut(o) and not filled(o) for o in obs_common)
    crossed = [o for o in placed if o.legacy_outcomes is not None]
    legacy = [(o, o.legacy_outcomes[p]) for o in crossed]

    def bucket_rates(key, buckets):
        groups: dict[str, list[OrderObs]] = defaultdict(list)
        for o in placed:
            groups[_bucket(key(o), buckets)].append(o)
        return {label: Rate(sum(filled(o) for o in groups[label]),
                            sum(not (cut(o) and not filled(o)) for o in groups[label]))
                for _, _, label in buckets if label in groups}

    km = kaplan_meier(
        (o.outcomes[p].first_fill_ts - o.inserted_ts, True) if filled(o)
        else (o.outcomes[p].cross_ts - o.inserted_ts, False) if resolved(o)
        else (o.end_ts - o.inserted_ts, False) if o.censored
        else (H, False)
        for o in placed)
    return FillSummary(
        strategy=name, policy=p,
        filled_any=len(any_), filled_full=len(full_),
        partial=sum(0 < o.outcomes[p].filled_qty < o.qty for o in placed),
        unfilled_observed=sum(not cut(o) and not filled(o) for o in placed),
        censored_unfilled=len(cens_unf), full_undetermined=len(full_und),
        fill_given_placed=Rate(len(any_), len(placed) - unknown),
        fill_given_placed_lower=Rate(len(any_), len(placed)),
        fill_given_placed_upper=Rate(len(any_) + unknown, len(placed)),
        fill_per_opportunity=Rate(len(any_), len(obs) - unknown),
        fill_common_support=Rate(common_any, len(obs_common) - common_cu),
        full_fill_given_placed=Rate(len(full_), len(placed) - len(full_und)),
        full_fill_per_opportunity=Rate(len(full_), len(obs) - len(full_und)),
        n_fills=len(fills),
        fills_by_mechanism=_counts(f.mechanism.value for f in fills),
        time_to_first_fill_ns=Dist.of(o.outcomes[p].first_fill_ts - o.inserted_ts for o in any_),
        time_to_complete_fill_ns=Dist.of(o.outcomes[p].complete_ts - o.inserted_ts for o in full_),
        km_first_fill=km,
        fill_by_ahead_orders=bucket_rates(lambda o: o.queue_ahead_orders0, AHEAD_ORDER_BUCKETS),
        fill_by_level_rank=bucket_rates(lambda o: o.level_rank0, LEVEL_RANK_BUCKETS),
        ahead_qty_at_fill=Dist.of(f.ahead_qty for f in fills if f.ahead_qty is not None),
        markouts={d: _markout_summary(d, fills, baseline) for d in report.markout_horizons_ns},
        fill_qty_by_evidence=_merge_qty(f.qty_by_evidence for f in fills),
        fills_by_evidence=_counts(max(f.qty_by_evidence, key=lambda k: Evidence[k]) for f in fills),
        resolved_by_cross=sum(resolved(o) for o in placed), crossed_unfilled=len(cross_unf),
        fills_after_cross=sum(f.after_cross for f in fills),
        legacy_filled_any=len(any_) + sum(lo.filled_qty > 0 and not filled(o) for o, lo in legacy)
        - sum(lo.filled_qty == 0 and filled(o) for o, lo in legacy),
        legacy_fills_after_cross=sum(sf.candidate.ts_ns > o.cross_ts for o, lo in legacy for sf in lo.fills),
        legacy_first_fill_after_cross=sum(lo.first_fill_ts is not None and lo.first_fill_ts > o.cross_ts
                                          for o, lo in legacy),
    )


def _markout_summary(d: int, fills: list[FillObs], baseline) -> MarkoutSummary:
    valid = [f for f in fills if f.markout_bps[d] is not None]
    values = [f.markout_bps[d] for f in valid]
    dist = Dist.of(values)
    qty = sum((f.qty for f in valid), Decimal(0))
    paired = [(f.markout_bps[d], baseline[(f.symbol, f.side, d)]) for f in valid
              if (f.symbol, f.side, d) in baseline]
    base = sum((b for _, b in paired), Decimal(0)) / len(paired) if paired else None
    adverse = -(sum((m for m, _ in paired), Decimal(0)) / len(paired) - base) if paired else None
    return MarkoutSummary(
        horizon_ns=d, n_fills=len(fills), n_valid=len(valid),
        missing=_counts(f.markout_missing[d] for f in fills if f.markout_bps[d] is None),
        mean=dist.mean, median=dist.median,
        qty_weighted_mean=sum((f.markout_bps[d] * f.qty for f in valid), Decimal(0)) / qty if qty else None,
        frac_negative=Rate(sum(v < 0 for v in values), len(values)),
        drift_baseline=base, adverse_selection=adverse, n_adverse=len(paired),
    )


def _counts(items) -> dict[str, int]:
    return dict(sorted(Counter(items).items()))


def _sum_by_evidence(parts) -> dict[str, Decimal]:
    out: dict[str, Decimal] = defaultdict(Decimal)
    for p in parts:
        out[p.evidence.name] += p.qty
    return dict(sorted(out.items()))


def _merge_qty(dicts) -> dict[str, Decimal]:
    out: dict[str, Decimal] = defaultdict(Decimal)
    for d in dicts:
        for k, v in d.items():
            out[k] += v
    return dict(sorted(out.items()))


def _evidence_stats(report: EvaluationReport, records) -> dict[str, EvidenceStats]:
    trades = report.evidence_mode == "trades"
    out = {}
    for name in report.strategies:
        n, count, qty = 0, Counter(), defaultdict(Decimal)
        for o in report.orders:
            if o.strategy != name or not o.eligible or o.stage != PLACED:
                continue
            order = records[(o.request.request_id, name)]
            for c in order.fills:
                if c.ts_ns >= o.inserted_ts + report.horizon_ns:
                    continue
                n += 1
                parts = c.attribution if trades else (EvidencePart(c.qty, c.evidence),)
                for cls in {p.evidence.name for p in parts}:
                    count[cls] += 1
                for p in parts:
                    qty[p.evidence.name] += p.qty
        out[name] = EvidenceStats(n, dict(sorted(count.items())), dict(sorted(qty.items())))
    return out


# --- comparison ----------------------------------------------------------------------------------------------


def comparison_rows(report: EvaluationReport, policy: str) -> list[dict]:
    """One flat row per strategy under `policy`: rates with their counts, so every number is auditable."""
    rows = []
    for name in report.strategies:
        lc, q, s = report.lifecycle[name], report.queue[name], report.summary[(name, policy)]
        row = {
            "strategy": name, "policy": policy, "evidence_mode": report.evidence_mode,
            "cross_rule": report.cross_rule, "horizon_ns": report.horizon_ns,
            "eligible": lc.eligible, "strategy_skips": lc.strategy_skips, "insertion_skips": lc.insertion_skips,
            "placed": lc.placed, "censored": lc.censored, "placement_rate": lc.placement_rate,
            "filled_any": s.filled_any, "filled_full": s.filled_full,
            "fill_given_placed": s.fill_given_placed, "fill_per_opportunity": s.fill_per_opportunity,
            "fill_common_support": s.fill_common_support, "censored_unfilled": s.censored_unfilled,
            "km_median_ttf_ns": s.km_first_fill.median_ns, "km_fill_prob": s.km_first_fill.fill_prob_at_horizon,
            "median_ttf_ns": s.time_to_first_fill_ns.median, "median_ahead_qty0": q.ahead_qty.median,
            "median_ahead_orders0": q.ahead_orders.median, "mean_edge_bps": q.edge_bps.mean, "n_fills": s.n_fills,
            "fill_qty_by_evidence": s.fill_qty_by_evidence,
            "fill_given_placed_lower": s.fill_given_placed_lower,
            "fill_given_placed_upper": s.fill_given_placed_upper,
            "crossed": lc.crossed, "resolved_by_cross": s.resolved_by_cross,
            "crossed_unfilled": s.crossed_unfilled, "fills_after_cross": s.fills_after_cross,
            "legacy_filled_any": s.legacy_filled_any, "legacy_fills_after_cross": s.legacy_fills_after_cross,
            "legacy_first_fill_after_cross": s.legacy_first_fill_after_cross,
        }
        for d, m in s.markouts.items():
            row[f"markout_{_label(d)}_n"] = m.n_valid
            row[f"markout_{_label(d)}_mean_bps"] = m.mean
            row[f"markout_{_label(d)}_median_bps"] = m.median
            row[f"markout_{_label(d)}_frac_neg"] = m.frac_negative
        rows.append(row)
    return rows


def format_report(report: EvaluationReport, markout_horizons_ns: Sequence[int] | None = None) -> str:
    """Plain-text summary: request accounting, lifecycle, then one fill/markout table per policy."""
    rc = report.request_counts
    shown = tuple(markout_horizons_ns or report.markout_horizons_ns)
    lines = [f"evidence {report.evidence_mode}, cross rule {report.cross_rule}, "
             f"horizon {report.horizon_ns / 1e9:g}s, latency {report.latency_ns / 1e6:g}ms",
             f"requests {rc['total']}: eligible {rc['eligible']}, common support {rc['common_support']}, "
             f"data gate {rc['data_gate']}, coverage {rc['coverage']}", "",
             f"{'strategy':<16}{'eligible':>9}{'skips':>7}{'ins.skip':>9}{'placed':>8}{'place rate':>11}"
             f"{'expired':>9}{'censored':>9}  skip reasons"]
    for name in report.strategies:
        lc = report.lifecycle[name]
        pr = lc.placement_rate.value
        lines.append(f"{name:<16}{lc.eligible:>9}{lc.strategy_skips:>7}{lc.insertion_skips:>9}{lc.placed:>8}"
                     f"{_f(pr):>11}{lc.expired:>9}{lc.censored:>9}  "
                     f"{ {**lc.strategy_skip_reasons, **lc.insertion_skip_reasons} or ''}")
    lines += ["", "crossed: an opposite-side order came to rest at or through our price before H "
              "(unexplained: no reduction could have reached us by then)",
              f"{'strategy':<16}{'placed':>8}{'crossed':>9}{'share':>8}{'unexplained':>13}"]
    for name in report.strategies:
        lc = report.lifecycle[name]
        lines.append(f"{name:<16}{lc.placed:>8}{lc.crossed:>9}{_f(Rate(lc.crossed, lc.placed).value):>8}"
                     f"{lc.inconsistent_crossings:>13}")
    lines += ["", "candidate volume by evidence (eligible placed orders, before H; n candidates)"]
    for name in report.strategies:
        ev = report.evidence.get(name)
        if ev is not None:
            lines.append(f"{name:<16}{ev.n_candidates:>7}  " + ", ".join(
                f"{k} {ev.candidates_by_evidence[k]} / {_f(v)}" for k, v in ev.qty_by_evidence.items()))
    if report.attribution:
        a = report.attribution
        lines += ["", f"trade attribution (whole recording): trades {a['trades']}",
                  f"  plausible executions by label: {a['reductions']}", f"  trade feed: {a['trade_feed']}"]
    for p in report.policies:
        lines += ["", f"policy {p}",
                  f"{'strategy':<16}{'filled':>7}{'P(f|placed)':>18}{'P(f|opp)':>18}{'P(f|common)':>18}"
                  f"{'KM med ms':>10}" + "".join(f"{'mk ' + _label(d) + ' n/mean/med/neg':>32}" for d in shown)]
        for name in report.strategies:
            s = report.summary[(name, p)]
            km = s.km_first_fill.median_ns
            row = (f"{name:<16}{s.filled_any:>7}{str(s.fill_given_placed):>18}{str(s.fill_per_opportunity):>18}"
                   f"{str(s.fill_common_support):>18}{'-' if km is None else f'{km / MS:.0f}':>10}")
            for d in shown:
                m = s.markouts[d]
                row += f"{f'{m.n_valid} / {_f(m.mean)} / {_f(m.median)} / {_f(m.frac_negative.value)}':>32}"
            lines.append(row)
        lines.append(f"  crossing under this policy ({report.cross_rule}); P(f|placed) bounds count every "
                     "unknown outcome as a non-fill / as a fill")
        for name in report.strategies:
            s = report.summary[(name, p)]
            lines.append(
                f"  {name:<14} unknown at crossing {s.resolved_by_cross} (unfilled {s.crossed_unfilled}), "
                f"censored unfilled {s.censored_unfilled}, bounds [{_f(s.fill_given_placed_lower.value)}, "
                f"{_f(s.fill_given_placed_upper.value)}]; 'ignore' would report {s.legacy_filled_any} filled, "
                f"{s.legacy_first_fill_after_cross} first filled after the crossing "
                f"({s.legacy_fills_after_cross} fills after it)")
    caveat = ("L3-only policies bracket the truth (conservative is always 0)." if report.evidence_mode == "l3_only"
              else "Trade mode: conservative = same-µs trade matches only; policies remain bounds.")
    cross_note = {
        "ignore": "Cross rule ignore (legacy): a crossed order keeps resting and later trades count.",
        "stop": "Cross rule stop: nothing after an order's crossing item counts; its outcome is unknown, not a non-fill.",
        "fill_at_cross": "Cross rule fill_at_cross: optimistic counts the crossing arrival and what follows; "
                         "conservative and balanced stop at the crossing.",
    }[report.cross_rule]
    lines += ["", "Markouts in bps vs the fill price, positive = favorable. Descriptive only: no significance "
              f"or profitability claim. {caveat}", cross_note,
              "Fills are counterfactual: the recorded market never contained these orders."]
    return "\n".join(lines)


def _f(v) -> str:
    return "-" if v is None else f"{float(v):.3f}"


def _label(ns: int) -> str:
    return f"{ns // MS}ms" if ns < 1_000 * MS else f"{ns / 1e9:g}s"
