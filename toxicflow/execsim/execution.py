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

Trade-based ranks (TRADE_ABSENT, TRADE_PLAUSIBLE, TRADE_CONFIRMED) need the
Phase 3 TradeIndex and are never produced here.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from .models import Evidence, FillCandidate, Mechanism, ReachEvent
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
    """Engine `on_reach` hook: records each ReachEvent as a FillCandidate on its order."""

    def __call__(self, order: SimulatedOrder, reach: ReachEvent, ev: BookEvent) -> None:
        order.add_fill(FillCandidate(
            ts_ns=reach.item_ts_ns,
            price=order.price,  # a maker fills at its own limit price
            qty=reach.volume,  # unclipped; settlement clips
            mechanism=reach.mechanism,
            evidence=l3_evidence(reach.mechanism, ev.kind),
            event_ref=reach.ref,
        ))


class FillStatus(str, Enum):
    UNFILLED = "UNFILLED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"


@dataclass(frozen=True, slots=True)
class SettledFill:
    """The part of one candidate that a policy counts: `qty` is clipped to what the order still needed."""

    candidate: FillCandidate
    qty: Decimal


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    policy: str
    filled_qty: Decimal
    status: FillStatus
    first_fill_ts: int | None  # ts of the first candidate that adds a positive amount
    complete_ts: int | None  # ts at which cumulative fill reaches the order quantity
    fills: tuple[SettledFill, ...]  # candidates that added a positive amount, in time order


def settle(order: SimulatedOrder, policy: str | Policy, until_ns: int | None = None,
           with_trades: bool = False) -> PolicyOutcome:
    """Cumulative fill of `order` under `policy` from candidates with ts < until_ns (§6.6).

    `until_ns` defaults to the order's expiry; candidates are only ever logged
    while the order is live, so the order's end is respected automatically.
    A skipped order (never inserted) settles as UNFILLED with no fills.
    """
    pol = POLICIES[policy] if isinstance(policy, str) else policy
    min_rank = pol.min_rank_with_trades if with_trades else pol.min_rank_l3_only
    cum, first, complete, fills = Decimal(0), None, None, []
    if min_rank is not None and order.inserted_ts is not None:
        cutoff = order.expiry_ts if until_ns is None else min(until_ns, order.expiry_ts)
        # candidates are appended in stream order; the stable sort only guards hand-built orders
        for c in sorted(order.fills, key=lambda f: f.ts_ns):
            if c.evidence < min_rank or c.ts_ns >= cutoff:
                continue
            add = min(c.qty, order.qty - cum)
            if add <= 0:
                break
            cum += add
            fills.append(SettledFill(c, add))
            first = c.ts_ns if first is None else first
            if cum == order.qty:
                complete = c.ts_ns
                break
    status = FillStatus.FILLED if cum == order.qty else FillStatus.PARTIAL if cum > 0 else FillStatus.UNFILLED
    return PolicyOutcome(pol.name, cum, status, first, complete, tuple(fills))
