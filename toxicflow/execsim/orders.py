"""One hypothetical passive order and its life cycle (design §8.6).

PENDING (decided, not yet in the queue) -> LIVE (queue tracked) -> ENDED.
A PENDING order that never reaches the queue ends through skip(); a LIVE order
ends through finish(). The order lives beside the historical book and is never
inserted into it.

Fills are kept as unclipped, labeled candidates. Policy settlement (per-policy
status, horizons, time to fill) belongs to FillSimulator; filled_qty() here is
only the clipped sum above an evidence threshold.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from toxicflow.kraken.book import CHECKSUM_LEVELS

from .models import (CrossEvent, EndReason, Evidence, FillCandidate, Mechanism, OrderIntent, PlacementRequest, Side,
                     SkipReason)
from .queue_tracker import QueueTracker


class OrderState(str, Enum):
    PENDING = "pending"
    LIVE = "live"
    ENDED = "ended"


@dataclass
class SimulatedOrder:
    order_id: str  # f"{request_id}/{strategy}"
    strategy: str
    request: PlacementRequest
    side: Side
    price: Decimal
    qty: Decimal
    decided_ts: int
    state: OrderState = OrderState.PENDING
    inserted_ts: int | None = None
    expiry_ts: int | None = None
    tracker: QueueTracker | None = None
    queue_ahead_qty0: Decimal = Decimal(0)
    queue_ahead_orders0: int = 0
    touch_qty0: Decimal = Decimal(0)
    level_rank0: int = 0
    max_level_rank: int = 0
    fills: list[FillCandidate] = field(default_factory=list)
    end_ts: int | None = None
    end_reason: EndReason | None = None
    skip_reason: SkipReason | None = None
    diag: Counter = field(default_factory=Counter)

    @classmethod
    def new(
        cls, request: PlacementRequest, strategy: str, intent: OrderIntent, qty: Decimal, decided_ts: int
    ) -> "SimulatedOrder":
        if qty <= 0:
            raise ValueError(f"order qty must be positive, got {qty}")
        if intent.side is not request.side:
            raise ValueError(f"intent side {intent.side} differs from request side {request.side}")
        return cls(
            order_id=f"{request.request_id}/{strategy}",
            strategy=strategy,
            request=request,
            side=intent.side,
            price=intent.price,
            qty=qty,
            decided_ts=decided_ts,
        )

    @property
    def book_side(self) -> str:
        return self.side.book_side

    @property
    def censored(self) -> bool:
        return self.end_reason is not None and self.end_reason.censored

    @property
    def checksum_covered(self) -> bool | None:
        """False if our level was ever ranked beyond the checksummed top levels (§6.8, §11.2):
        its queue was then not validated by Kraken's checksum. None if never placed."""
        if self.inserted_ts is None:
            return None
        return self.max_level_rank <= CHECKSUM_LEVELS

    @property
    def cross(self) -> CrossEvent | None:
        """The first opposite-side order that came to rest at or through our price while we were
        LIVE (the tracker only sees events of a live order). None if never crossed or never placed."""
        return self.tracker.cross if self.tracker is not None else None

    @property
    def inconsistent_crossing(self) -> bool | None:
        """Crossed with no LEVEL or INTERCEPT candidate up to the crossing time (§6.8): the market
        went through our price without any reduction that could have reached us, so the crossing
        is not explained by a sweep. None if never crossed."""
        cross = self.cross
        if cross is None:
            return None
        return not any(f.mechanism is not Mechanism.CROSS_ARRIVAL and f.ts_ns <= cross.ts_ns for f in self.fills)

    def start(
        self,
        inserted_ts: int,
        ahead: Sequence[tuple[str, Decimal]],
        rank: int,
        touch_qty: Decimal,
        horizon_ns: int,
        increase_policy: str = "keep",
    ) -> None:
        """Enter the queue behind every order in `ahead` (SimL3Book.level_orders at insertion)."""
        self._require(OrderState.PENDING, "start")
        if horizon_ns <= 0:
            raise ValueError(f"horizon_ns must be positive, got {horizon_ns}")
        self.tracker = QueueTracker(self.side, self.price, ahead, increase_policy)
        self.state = OrderState.LIVE
        self.inserted_ts = inserted_ts
        self.expiry_ts = inserted_ts + horizon_ns
        self.queue_ahead_qty0 = self.tracker.initial_ahead_qty
        self.queue_ahead_orders0 = self.tracker.initial_ahead_orders
        self.touch_qty0 = touch_qty
        self.level_rank0 = self.max_level_rank = rank

    def add_fill(self, candidate: FillCandidate) -> None:
        self._require(OrderState.LIVE, "add_fill")
        if candidate.price != self.price:
            raise ValueError(f"candidate price {candidate.price} is not the limit price {self.price}")
        self.fills.append(candidate)

    def finish(self, reason: EndReason, end_ts: int) -> None:
        """End a LIVE order. Censored orders keep their fills; their status is a lower bound."""
        self._require(OrderState.LIVE, "finish")
        self.state = OrderState.ENDED
        self.end_reason = reason
        self.end_ts = end_ts
        self.tracker.invalidate(reason)
        self.diag.update(self.tracker.diag)

    def skip(self, reason: SkipReason) -> None:
        """A PENDING order that never reached the queue: counted as skipped, never as unfilled."""
        self._require(OrderState.PENDING, "skip")
        self.state = OrderState.ENDED
        self.skip_reason = reason

    def filled_qty(self, min_rank: Evidence | int | None) -> Decimal:
        """Cumulative fill counting candidates with evidence >= min_rank, clipped to qty.
        None means no candidate qualifies (conservative in L3-only mode)."""
        if min_rank is None:
            return Decimal(0)
        total = sum((f.qty for f in self.fills if f.evidence >= min_rank), Decimal(0))
        return min(self.qty, total)

    def remaining_qty(self, min_rank: Evidence | int | None) -> Decimal:
        return self.qty - self.filled_qty(min_rank)

    def _require(self, state: OrderState, action: str) -> None:
        if self.state is not state:
            raise RuntimeError(f"{self.order_id}: cannot {action} in state {self.state.value}")
