"""Enums and value types shared by the simulator components (design §8.2).

All prices and quantities are Decimal; all times are integer nanoseconds.
BookEvent, ApplyInfo and SyncStatus live in sim_book.py.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum, IntEnum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def book_side(self) -> str:
        return "bids" if self is Side.BUY else "asks"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    def better(self, a: Decimal, b: Decimal) -> bool:
        """True if price `a` has priority over price `b` on this side (higher bid, lower ask)."""
        return self.sign * (a - b) > 0


class Evidence(IntEnum):
    """How strongly a reachable reduction is believed to be an execution (§6.5.2)."""

    TRADE_ABSENT = 0
    L3_CROSS_ARRIVAL = 1
    L3_DELETE_AMBIGUOUS = 2
    L3_MODIFY_DOWN = 3
    TRADE_PLAUSIBLE = 4
    TRADE_CONFIRMED = 5


class Mechanism(str, Enum):
    LEVEL = "level"  # reduction at our price with nothing ahead of us
    INTERCEPT = "intercept"  # reduction at the touch, at a price worse than ours
    CROSS_ARRIVAL = "cross_arrival"  # opposite-side add at a price that crosses ours


class EndReason(str, Enum):
    """How a LIVE order ended (§6.7). Every reason except the first two is a censor."""

    HORIZON = "HORIZON"
    ALL_POLICIES_FILLED = "ALL_POLICIES_FILLED"
    CENSORED_DESYNC = "CENSORED_DESYNC"
    CENSORED_RESNAPSHOT = "CENSORED_RESNAPSHOT"
    CENSORED_RECV_GAP = "CENSORED_RECV_GAP"
    CENSORED_SESSION_RESTART = "CENSORED_SESSION_RESTART"
    CENSORED_TRUNCATED = "CENSORED_TRUNCATED"  # raised by the §7.3 pseudocode, missing from the §6.7 table
    CENSORED_END_OF_DATA = "CENSORED_END_OF_DATA"
    OUT_OF_DEPTH = "OUT_OF_DEPTH"

    @property
    def censored(self) -> bool:
        return self not in (EndReason.HORIZON, EndReason.ALL_POLICIES_FILLED)


class SkipReason(str, Enum):
    """Why no order was created, or why a PENDING order never reached the book."""

    NO_TOUCH = "NO_TOUCH"
    ONE_SIDED_BOOK = "ONE_SIDED_BOOK"
    SPREAD_TOO_NARROW = "SPREAD_TOO_NARROW"
    INVALID_PRICE = "INVALID_PRICE"
    UNSYNCED = "UNSYNCED"
    WARMUP = "WARMUP"
    DATA_GAP = "DATA_GAP"
    NO_DATA = "NO_DATA"
    CROSSED_AT_INSERT = "CROSSED_AT_INSERT"
    OUT_OF_DEPTH = "OUT_OF_DEPTH"  # level not visible at insertion (§7.3 insert)


@dataclass(frozen=True, slots=True)
class PlacementRequest:
    request_id: str
    symbol: str
    side: Side
    ts_ns: int


@dataclass(frozen=True, slots=True)
class OrderIntent:
    side: Side
    price: Decimal


@dataclass(frozen=True, slots=True)
class Skip:
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class LevelSummary:
    price: Decimal
    qty: Decimal
    n_orders: int


@dataclass(frozen=True, slots=True)
class BookView:
    """Everything a strategy may see: an immutable copy of the top of the book at a decision boundary."""

    symbol: str
    ts_ns: int  # the decision time; the view holds only items with timestamp < ts_ns
    tick: Decimal
    bids: tuple[LevelSummary, ...]  # best first, top view_depth levels
    asks: tuple[LevelSummary, ...]
    book_timestamp: str | None = None  # raw item timestamp of the last item applied (for audits)

    @property
    def best_bid(self) -> Decimal | None:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Decimal | None:
        return self.asks[0].price if self.asks else None

    def best(self, side: Side) -> Decimal | None:
        """Best bid for BUY, best ask for SELL: the price that side would join."""
        return self.best_bid if side is Side.BUY else self.best_ask


@dataclass(frozen=True, slots=True)
class ReachEvent:
    """A historical event that could have filled our order. A candidate, not a fill."""

    ref: tuple[int | None, int]  # (position.seq, event_index) of the BookEvent
    item_ts_ns: int | None  # engine clock for the item; None when the caller did not supply it
    event_price: Decimal  # where the historical volume was removed (or arrived, for CROSS_ARRIVAL)
    volume: Decimal  # historical quantity, unclipped
    mechanism: Mechanism


@dataclass(frozen=True, slots=True)
class CrossEvent:
    """The first opposite-side order that came to rest at or through our limit price while we were
    live (buy: an ask at or below our price; sell: a bid at or above it).

    From this moment the recording is no longer a world that could have contained our order: a
    real order at our price would have traded with the arrival, or (post-only) the arrival would
    have been rejected. What happens to the order next is decided by the crossing rule at
    settlement (execution.CROSS_RULES); the tracker only records the fact.
    """

    ref: tuple[int | None, int]  # (position.seq, event_index) of the crossing BookEvent
    ts_ns: int | None  # engine clock of its item
    price: Decimal  # where the opposite order came to rest
    qty: Decimal  # its quantity
    ahead_qty: Decimal  # our queue ahead at that moment (non-zero: the historical book itself was crossed)
    kind: str  # 'add', or 'modify' for a resting order re-priced through us (never a fill candidate)


@dataclass(frozen=True, slots=True)
class EvidencePart:
    """A slice of one candidate's volume with a single trade-attribution label (§6.5.4: a partially
    covered reduction counts only for the covered part)."""

    qty: Decimal
    evidence: Evidence
    trade_ids: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class FillCandidate:
    ts_ns: int | None
    price: Decimal  # our limit price: a maker fills at its own price
    qty: Decimal  # unclipped historical volume
    mechanism: Mechanism
    evidence: Evidence  # L3-only label (rank 1 to 3), always set
    event_ref: tuple[int | None, int]
    trade_ids: tuple[int, ...] = ()
    # trade-attributed slices summing to qty; None when the replay had no trade data
    attribution: tuple[EvidencePart, ...] | None = None
