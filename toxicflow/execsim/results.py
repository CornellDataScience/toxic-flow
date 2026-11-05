"""In-memory replay results. ResultWriter (JSONL output, §11) is not built yet."""

from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal

from .models import BookView, PlacementRequest, ReachEvent, SkipReason
from .orders import SimulatedOrder


DISCONTINUITY_KINDS = frozenset({"SESSION_RESTART", "RECV_GAP", "TRUNCATED", "DESYNC", "RESNAPSHOT"})


@dataclass(frozen=True, slots=True)
class QualityEvent:
    """A discontinuity or data-quality observation, in stream order.

    One break in the data usually produces several records: a checksum failure is
    DESYNC and, when the book is reloaded, RESNAPSHOT; an hourly file boundary is
    SESSION_RESTART and then RESNAPSHOT. All records are kept for diagnostics, and
    `incident` groups them: a discontinuity raised while the symbol is synced opens
    a new incident, one raised while it is still broken joins the open one, and the
    next sync closes it. Count incidents as distinct (symbol, incident) pairs, never
    as records. Non-discontinuity kinds have incident None.
    """

    ts_ns: int  # the symbol's guarded clock when it was seen
    symbol: str
    kind: str  # DISCONTINUITY_KINDS, SNAPSHOT_FIRST, SNAPSHOT_CHECKSUM_FAIL, TS_REGRESSION, TS_MISSING, ...
    seq: int | None  # StreamRecord.seq where it was seen
    recv_ns: int | None
    detail: dict = field(default_factory=dict)
    incident: int | None = None  # per-symbol incident number (discontinuities only)


@dataclass(frozen=True, slots=True)
class SkipRecord:
    """A (request, strategy) pair for which no SimulatedOrder was created."""

    request: PlacementRequest
    strategy: str
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class ReachRecord:
    """A ReachEvent logged for an order. Not a fill: it carries no evidence label yet.

    Every ReachRecord comes from an item whose checksum passed (I7); events of a failing
    item are never routed, so unvalidated evidence cannot appear here.
    """

    order_id: str
    strategy: str
    symbol: str
    reach: ReachEvent
    ahead_qty: Decimal  # queue ahead when it was seen (0 for LEVEL and CROSS_ARRIVAL)
    checksum_covered: bool  # event price within the checksummed top levels of its side after the item


@dataclass
class ReplayResult:
    orders: list[SimulatedOrder]  # every order created, ended or skipped at insertion, in creation order
    skips: list[SkipRecord]
    reaches: list[ReachRecord]
    quality: list[QualityEvent]
    views: dict[str, BookView]  # request_id -> the one view every strategy saw
    counters: dict[str, Counter]  # per symbol
