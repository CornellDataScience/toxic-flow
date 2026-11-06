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
"""

import bisect
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from .config import SimConfig
from .execution import POLICY_NAMES, FillRecorder, PolicyOutcome, settle
from .markouts import (DEFAULT_MARKOUT_HORIZONS_NS, MidSample, drift_bps, edge_bps, sample_mids, signed_markout_bps,
                       view_mid)
from .models import EndReason, Evidence, Mechanism, PlacementRequest, Side, SkipReason
from .orders import OrderState
from .replay import HistoricalReplayEngine
from .results import DISCONTINUITY_KINDS, ReplayResult, SkipRecord
from .strategies import OrderPlacementStrategy
from .timeutil import MS

ELIGIBLE, DATA_GATE, COVERAGE = "ELIGIBLE", "DATA_GATE", "COVERAGE"
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


# --- analysis ----------------------------------------------------------------------------------------------------


def evaluate(cfg: SimConfig, strategies: Sequence[OrderPlacementStrategy], schedule: Iterable[PlacementRequest],
             files: Sequence[Path], horizon_ns: int | None = None,
             markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
             policies: Sequence[str] = POLICY_NAMES) -> tuple[ReplayResult, EvaluationReport]:
    """Replay with fill recording, sample the mids the markouts need (second pass), and analyze."""
    eng = HistoricalReplayEngine(cfg, strategies, schedule, on_reach=FillRecorder())
    result = eng.run(files)
    times = required_mid_times(result, cfg, horizon_ns, markout_horizons_ns, policies)
    mids = sample_mids(cfg, times, files=files)
    report = analyze(result, cfg, mids, horizon_ns, markout_horizons_ns, policies,
                     strategies=[s.name for s in strategies])
    return result, report


def required_mid_times(result: ReplayResult, cfg: SimConfig, horizon_ns: int | None = None,
                       markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
                       policies: Sequence[str] = POLICY_NAMES) -> set[tuple[str, int]]:
    """(symbol, ts) pairs analyze() needs mids for: every settled fill + D, and every eligible request T + D."""
    report = analyze(result, cfg, None, horizon_ns, markout_horizons_ns, policies)
    times = {(f.symbol, f.ts_ns + d) for f in report.fills for d in report.markout_horizons_ns}
    times |= {(r.request.symbol, r.request.ts_ns + d) for r in report.requests if r.status == ELIGIBLE
              for d in report.markout_horizons_ns}
    return times


def analyze(result: ReplayResult, cfg: SimConfig, mids: dict[tuple[str, int], MidSample] | None = None,
            horizon_ns: int | None = None, markout_horizons_ns: Sequence[int] = DEFAULT_MARKOUT_HORIZONS_NS,
            policies: Sequence[str] = POLICY_NAMES, strategies: Sequence[str] | None = None) -> EvaluationReport:
    """Metrics for every strategy in `result` at analysis horizon H = `horizon_ns` (default: the run's).

    `result` must come from a replay with FillRecorder as `on_reach`; without `mids`, every markout
    is missing ('NOT_SAMPLED').
    """
    H = cfg.horizon_ns if horizon_ns is None else horizon_ns
    if not 0 < H <= cfg.horizon_ns:
        raise ValueError(f"analysis horizon must be in (0, {cfg.horizon_ns}], got {horizon_ns}")
    horizons = tuple(markout_horizons_ns)
    if any(d <= 0 for d in horizons) or len(set(horizons)) != len(horizons):
        raise ValueError(f"markout horizons must be positive and unique, got {horizons}")
    _check_fill_recording(result)

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
            obs = _order_obs(records[(r.request.request_id, name)], r, H, policies)
            orders.append(obs)
            if obs.eligible and obs.stage == PLACED:
                fills.extend(_fill_obs(obs, policies, reach_ahead, mids, horizons))

    baseline = _drift_baseline(requests, mids, horizons)
    report = EvaluationReport(H, cfg.latency_ns, names, tuple(policies), horizons, requests, orders, fills, {}, {},
                              {}, {})
    _summarize(report, baseline)
    return report


def _check_fill_recording(result: ReplayResult) -> None:
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


def _order_obs(rec, r: RequestObs, H: int, policies: Sequence[str]) -> OrderObs:
    eligible = r.status == ELIGIBLE
    if isinstance(rec, SkipRecord):  # SkipRecord: strategy skip (or the data gate for every strategy)
        return OrderObs(r.request, rec.strategy, eligible, SKIPPED, rec.reason)
    o = rec
    if o.inserted_ts is None:
        return OrderObs(r.request, o.strategy, eligible, INSERT_SKIPPED, o.skip_reason, o.order_id, o.price, o.qty)
    censored = o.end_reason.censored and o.end_ts < o.inserted_ts + H
    edge = edge_bps(o.side, o.price, r.decision_mid) if r.decision_mid is not None else None
    outcomes = {p: settle(o, p, until_ns=o.inserted_ts + H) for p in policies}
    return OrderObs(r.request, o.strategy, eligible, PLACED, None, o.order_id, o.price, o.qty, o.inserted_ts,
                    o.end_ts, o.end_reason, censored, o.queue_ahead_qty0, o.queue_ahead_orders0, o.level_rank0,
                    o.touch_qty0, edge, outcomes)


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
                               reach_ahead.get((o.order_id, c.event_ref)), marks, why))
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

    any_ = [o for o in placed if filled(o)]
    full_ = [o for o in placed if full(o)]
    cens_unf = [o for o in placed if o.censored and not filled(o)]
    full_und = [o for o in placed if o.censored and not full(o)]
    obs_common = [o for o in obs if o.request.request_id in common]
    common_any = sum(o.stage == PLACED and filled(o) for o in obs_common)
    common_cu = sum(o.stage == PLACED and o.censored and not filled(o) for o in obs_common)

    def bucket_rates(key, buckets):
        groups: dict[str, list[OrderObs]] = defaultdict(list)
        for o in placed:
            groups[_bucket(key(o), buckets)].append(o)
        return {label: Rate(sum(filled(o) for o in groups[label]),
                            sum(not (o.censored and not filled(o)) for o in groups[label]))
                for _, _, label in buckets if label in groups}

    km = kaplan_meier(
        (o.outcomes[p].first_fill_ts - o.inserted_ts, True) if filled(o)
        else (o.end_ts - o.inserted_ts, False) if o.censored
        else (H, False)
        for o in placed)
    return FillSummary(
        strategy=name, policy=p,
        filled_any=len(any_), filled_full=len(full_),
        partial=sum(0 < o.outcomes[p].filled_qty < o.qty for o in placed),
        unfilled_observed=sum(not o.censored and not filled(o) for o in placed),
        censored_unfilled=len(cens_unf), full_undetermined=len(full_und),
        fill_given_placed=Rate(len(any_), len(placed) - len(cens_unf)),
        fill_given_placed_lower=Rate(len(any_), len(placed)),
        fill_given_placed_upper=Rate(len(any_) + len(cens_unf), len(placed)),
        fill_per_opportunity=Rate(len(any_), len(obs) - len(cens_unf)),
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


# --- comparison ----------------------------------------------------------------------------------------------


def comparison_rows(report: EvaluationReport, policy: str) -> list[dict]:
    """One flat row per strategy under `policy`: rates with their counts, so every number is auditable."""
    rows = []
    for name in report.strategies:
        lc, q, s = report.lifecycle[name], report.queue[name], report.summary[(name, policy)]
        row = {
            "strategy": name, "policy": policy, "horizon_ns": report.horizon_ns,
            "eligible": lc.eligible, "strategy_skips": lc.strategy_skips, "insertion_skips": lc.insertion_skips,
            "placed": lc.placed, "censored": lc.censored, "placement_rate": lc.placement_rate,
            "filled_any": s.filled_any, "filled_full": s.filled_full,
            "fill_given_placed": s.fill_given_placed, "fill_per_opportunity": s.fill_per_opportunity,
            "fill_common_support": s.fill_common_support, "censored_unfilled": s.censored_unfilled,
            "km_median_ttf_ns": s.km_first_fill.median_ns, "km_fill_prob": s.km_first_fill.fill_prob_at_horizon,
            "median_ttf_ns": s.time_to_first_fill_ns.median, "median_ahead_qty0": q.ahead_qty.median,
            "median_ahead_orders0": q.ahead_orders.median, "mean_edge_bps": q.edge_bps.mean, "n_fills": s.n_fills,
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
    lines = [f"horizon {report.horizon_ns / 1e9:g}s, latency {report.latency_ns / 1e6:g}ms",
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
    lines += ["", "Markouts in bps vs the fill price, positive = favorable. Descriptive only: no significance "
              "or profitability claim; L3-only policies bracket the truth (conservative is always 0)."]
    return "\n".join(lines)


def _f(v) -> str:
    return "-" if v is None else f"{float(v):.3f}"


def _label(ns: int) -> str:
    return f"{ns // MS}ms" if ns < 1_000 * MS else f"{ns / 1e9:g}s"
