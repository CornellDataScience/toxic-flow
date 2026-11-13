"""Evidence labels, fill policies and settlement, L3-only mode (design §6.5.2, §6.5.3, §6.6, §8.7).

The replay engine logs every ReachEvent (a historical reduction that could have
filled one of our orders). This module turns them into labeled FillCandidates
and settles them per policy. It does not change the order life cycle: orders
still run to their horizon, and the engine is untouched.

Usage: pass a FillRecorder as the engine's `on_reach` hook, then call settle()
on each ended order.

    eng = HistoricalReplayEngine(cfg, strategies, schedule, on_reach=FillRecorder())

Evidence in L3-only mode is a function of the event alone (computed once per
event, never per order), so a reduction that reaches several of our orders gets
the same label for each:

    CROSS_ARRIVAL               -> L3_CROSS_ARRIVAL    (rank 1)
    LEVEL / INTERCEPT, delete   -> L3_DELETE_AMBIGUOUS (rank 2)
    LEVEL / INTERCEPT, modify   -> L3_MODIFY_DOWN      (rank 3)

Trade-based ranks (TRADE_ABSENT, TRADE_PLAUSIBLE, TRADE_CONFIRMED) come from trades.py:
give FillRecorder a TradeAttribution and each candidate also carries `attribution`, its
volume split into trade-labeled slices. `evidence` always keeps the L3 label, so one replay
can be settled both ways: settle(..., with_trades=False) is the L3-only bracket,
with_trades=True applies the trade thresholds to the slices.

Crossing rules (CROSS_RULES). An order is *crossed* when an opposite-side order comes to rest at
or through its limit price while it is live (SimulatedOrder.cross). A real order at that price
would have traded with the arrival, or the arrival (if post-only) would have been rejected, so
the recording after that moment is not a world that contained our order. Candidates are logged
the same way whatever the rule; the rule only chooses which of them settlement may count:

    ignore          legacy: the order keeps resting; every later candidate counts by its evidence
                    (a later trade at a worse price can then be booked as our fill)
    stop            the order is observed up to the crossing item and no further: candidates of
                    later items, and every crossing arrival, are not counted. An order that was
                    not complete by then is `resolved_by_cross`: its outcome is unknown, which is
                    neither a fill nor an observed non-fill. No claim is made about the crossing.
    fill_at_cross   the crossing arrival is a fill candidate at its own event time with its own
                    evidence (L3_CROSS_ARRIVAL, rank 1) and its own quantity; nothing is added.
                    A policy that accepts that evidence (optimistic) counts it and keeps the
                    remainder resting, as in `ignore`. A policy that does not accept it
                    (conservative, balanced) cannot say what happened at the crossing and stops
                    there, as in `stop`.

Candidates from the crossing item itself still count under every rule: an aggressive order that
sweeps our level and rests its remainder produces the fills and the crossing in one item.
Under every rule conservative <= balanced <= optimistic, because each policy applies a rank
threshold to a candidate set that only grows from conservative to optimistic.
REPORTING_CROSS_RULE is what the runner and reports default to; settle() itself defaults to
`ignore` so existing callers and stored results keep their meaning.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .models import Evidence, EvidencePart, FillCandidate, Mechanism, ReachEvent
from .orders import SimulatedOrder
from .sim_book import BookEvent


@dataclass(frozen=True, slots=True)
class Policy:
    """A fill policy is a minimum evidence rank; None means no candidate qualifies."""

    name: str
    min_rank_l3_only: int | None
    min_rank_with_trades: int | None


POLICIES = {
    "conservative": Policy("conservative", None, Evidence.TRADE_CONFIRMED),
    "balanced": Policy("balanced", Evidence.L3_MODIFY_DOWN, Evidence.TRADE_PLAUSIBLE),
    "optimistic": Policy("optimistic", Evidence.L3_CROSS_ARRIVAL, Evidence.L3_CROSS_ARRIVAL),
}
POLICY_NAMES = tuple(POLICIES)  # lower bound first

CROSS_RULES = ("ignore", "stop", "fill_at_cross")
# Headline default: `stop` asserts nothing about what happened at a crossing. `fill_at_cross`
# gives the same conservative and balanced numbers and differs only in optimistic.
REPORTING_CROSS_RULE = "stop"


def l3_evidence(mechanism: Mechanism, event_kind: str) -> Evidence:
    """Evidence rank of a reachable reduction from L3 data alone (§6.5.2)."""
    if mechanism is Mechanism.CROSS_ARRIVAL:
        return Evidence.L3_CROSS_ARRIVAL
    if event_kind == "delete":
        return Evidence.L3_DELETE_AMBIGUOUS
    if event_kind == "modify":
        return Evidence.L3_MODIFY_DOWN
    raise ValueError(f"no L3 evidence for a {mechanism.value} reach from a {event_kind!r} event")


class FillRecorder:
    """Engine `on_reach` hook: records each ReachEvent as a FillCandidate on its order.

    `attribution` (a trades.TradeAttribution built from the same L3 files) adds the trade slices.
    """

    def __init__(self, attribution=None):
        self.attribution = attribution

    def __call__(self, order: SimulatedOrder, reach: ReachEvent, ev: BookEvent) -> None:
        parts = None
        if self.attribution is not None:
            parts = self.attribution.parts_for(reach.ref, ev, reach.volume, reach.mechanism)
        order.add_fill(FillCandidate(
            ts_ns=reach.item_ts_ns,
            price=order.price,  # a maker fills at its own limit price
            qty=reach.volume,  # unclipped; settlement clips
            mechanism=reach.mechanism,
            evidence=l3_evidence(reach.mechanism, ev.kind),
            event_ref=reach.ref,
            trade_ids=tuple(i for p in parts or () for i in p.trade_ids),
            attribution=parts,
        ))


class FillStatus(str, Enum):
    UNFILLED = "UNFILLED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"


@dataclass(frozen=True, slots=True)
class SettledFill:
    """The part of one candidate that a policy counts: `qty` is clipped to what the order still needed.

    `by_evidence` splits `qty` over the evidence that justified it, strongest evidence used first.
    """

    candidate: FillCandidate
    qty: Decimal
    by_evidence: tuple[EvidencePart, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    policy: str
    filled_qty: Decimal
    status: FillStatus
    first_fill_ts: int | None  # ts of the first candidate that adds a positive amount
    complete_ts: int | None  # ts at which cumulative fill reaches the order quantity
    fills: tuple[SettledFill, ...]  # candidates that added a positive amount, in time order
    cross_rule: str = "ignore"
    cross_ts: int | None = None  # time of the first crossing inside the settled window (any rule)
    # this policy stopped observing at the crossing with the order still incomplete: outcome unknown
    resolved_by_cross: bool = False


def settle(order: SimulatedOrder, policy: str | Policy, until_ns: int | None = None,
           with_trades: bool = False, cross_rule: str = "ignore") -> PolicyOutcome:
    """Cumulative fill of `order` under `policy` from candidates with ts < until_ns (§6.6).

    `until_ns` defaults to the order's expiry; candidates are only ever logged
    while the order is live, so the order's end is respected automatically.
    A skipped order (never inserted) settles as UNFILLED with no fills.

    with_trades=False: a candidate counts in full if its L3 label meets the L3-only threshold.
    with_trades=True: only its trade slices that meet the with-trades threshold count; every
    candidate must carry `attribution` (replay with FillRecorder(attribution)).

    `cross_rule` (CROSS_RULES, see the module docstring) decides what a crossing of the order's
    price inside the window does to the candidates that may count.
    """
    if cross_rule not in CROSS_RULES:
        raise ValueError(f"cross_rule must be one of {CROSS_RULES}, got {cross_rule!r}")
    pol = POLICIES[policy] if isinstance(policy, str) else policy
    min_rank = pol.min_rank_with_trades if with_trades else pol.min_rank_l3_only
    cum, first, complete, fills = Decimal(0), None, None, []
    cross_ts, stops = None, False
    if order.inserted_ts is not None:
        cutoff = order.expiry_ts if until_ns is None else min(until_ns, order.expiry_ts)
        cross = order.cross
        if cross is not None and cross.ts_ns < cutoff:
            cross_ts = cross.ts_ns
            accepts_crossing = min_rank is not None and min_rank <= Evidence.L3_CROSS_ARRIVAL
            stops = cross_rule == "stop" or (cross_rule == "fill_at_cross" and not accepts_crossing)
    if min_rank is not None and order.inserted_ts is not None:
        # candidates are appended in stream order; the stable sort only guards hand-built orders
        for c in sorted(order.fills, key=lambda f: f.ts_ns):
            if c.ts_ns >= cutoff:
                continue
            if stops and (c.ts_ns > cross_ts or c.mechanism is Mechanism.CROSS_ARRIVAL):
                continue  # beyond what this policy observes of a crossed order
            slices = _qualifying(c, min_rank, with_trades)
            usable = sum((p.qty for p in slices), Decimal(0))
            if usable <= 0:
                continue
            add = min(usable, order.qty - cum)
            if add <= 0:
                break
            cum += add
            fills.append(SettledFill(c, add, _allocate(slices, add)))
            first = c.ts_ns if first is None else first
            if cum == order.qty:
                complete = c.ts_ns
                break
    status = FillStatus.FILLED if cum == order.qty else FillStatus.PARTIAL if cum > 0 else FillStatus.UNFILLED
    return PolicyOutcome(pol.name, cum, status, first, complete, tuple(fills), cross_rule, cross_ts,
                         stops and cum < order.qty)


def _qualifying(c: FillCandidate, min_rank: int, with_trades: bool) -> list[EvidencePart]:
    if not with_trades:
        return [EvidencePart(c.qty, c.evidence, c.trade_ids)] if c.evidence >= min_rank else []
    if c.attribution is None:
        raise ValueError(f"candidate {c.event_ref} has no trade attribution: replay with FillRecorder(attribution)")
    return [p for p in c.attribution if p.evidence >= min_rank]


def _allocate(slices: list[EvidencePart], qty: Decimal) -> tuple[EvidencePart, ...]:
    """Split a settled quantity over qualifying slices, strongest evidence first (deterministic)."""
    out, left = [], qty
    for p in sorted(slices, key=lambda p: -p.evidence):
        take = min(p.qty, left)
        if take > 0:
            out.append(EvidencePart(take, p.evidence, p.trade_ids))
            left -= take
    return tuple(out)
