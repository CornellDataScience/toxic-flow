"""L3Book that reports what each applied event did, and serves read-only views.

SimL3Book.apply() is the only writer of book state. Per item it fills
`last_events` (one BookEvent per add/modify/delete, computed from the state
immediately before and after that event) and `last_info`. Hypothetical orders
are never inserted into this book.

Events are kept even when the item's checksum fails; the caller must discard
them unless `last_info.result is True`.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum

from toxicflow.kraken.book import L3Book, PairInfo

from .reader import StreamRecord

BOOK_SIDES = ("bids", "asks")


class SyncStatus(str, Enum):
    UNSYNCED = "unsynced"  # no snapshot yet
    SYNCED = "synced"
    CHECKSUM_FAILED = "checksum_failed"  # unsynced until the next snapshot


@dataclass(frozen=True, slots=True)
class BookEvent:
    kind: str  # add | modify | delete, as sent
    side: str  # "bids" | "asks"
    order_id: str
    price: Decimal  # the event's limit_price
    qty: Decimal  # the event's order_qty
    qty_before: Decimal | None  # None if the ID was not on the book
    qty_after: Decimal | None  # None if the order is not on the book afterwards
    price_before: Decimal | None
    price_after: Decimal | None
    queue_index_before: int | None  # 0 = front of its level; None if unknown or not tracked
    queue_index_after: int | None
    requeued: bool  # (re)appended at the back of a level, i.e. priority lost
    dropped_by_truncation: bool  # the order was forgotten by depth truncation at the end of the item
    best_bid_before: Decimal | None
    best_ask_before: Decimal | None
    best_bid_after: Decimal | None
    best_ask_after: Decimal | None
    item_timestamp: str | None  # the clock (item-level string, unparsed)
    event_timestamp: str | None  # the event's own string (metadata, may be old)
    event_index: int  # index within the item, bids first then asks
    position: StreamRecord | None

    @property
    def noop(self) -> bool:
        return self.qty_before is None and self.qty_after is None

    @property
    def reduction(self) -> Decimal:
        """Quantity removed from the order (0 if it grew, was added, or the event was a no-op)."""
        if self.qty_before is None:
            return Decimal(0)
        return max(self.qty_before - (self.qty_after or Decimal(0)), Decimal(0))


@dataclass(frozen=True, slots=True)
class ApplyInfo:
    msg_type: str
    result: bool | None  # apply()'s return value
    snapshot_loaded: bool
    book_reset: bool
    checksum_failed: bool
    ignored_unsynced: bool
    n_events: int
    truncated: tuple[tuple[str, Decimal, int], ...]  # (side, price, n_orders) levels dropped by depth


class ReadOnlyBookView:
    """Query-only window onto a SimL3Book. Everything returned is a copy or an immutable value."""

    __slots__ = ("_book",)

    def __init__(self, book: "SimL3Book"):
        self._book = book

    @property
    def symbol(self) -> str:
        return self._book.symbol

    @property
    def synced(self) -> bool:
        return self._book.sync_status == SyncStatus.SYNCED

    def best_bid(self) -> Decimal | None:
        return self._book.best_bid()

    def best_ask(self) -> Decimal | None:
        return self._book.best_ask()

    def mid(self) -> Decimal | None:
        return self._book.mid()

    def level_qty(self, side: str, price: Decimal) -> Decimal:
        return self._book.level_qty(side, price)

    def level_orders(self, side: str, price: Decimal) -> tuple[tuple[str, Decimal], ...]:
        return self._book.level_orders(side, price)

    def levels(self, side: str, n: int | None = None) -> tuple[tuple[Decimal, Decimal, int], ...]:
        """(price, total qty, n_orders) for the best `n` levels, best first."""
        return self._book.levels(side, n)


class SimL3Book(L3Book):
    def __init__(self, pair: PairInfo, depth: int, track_queue: bool = True):
        """`track_queue` computes per-event queue indexes (O(level size) per modify/delete)."""
        self.track_queue = track_queue
        super().__init__(pair, depth)

    def reset(self) -> None:
        # called from the base __init__ and on every snapshot
        super().reset()
        self.sync_status = SyncStatus.UNSYNCED
        self.last_events: tuple[BookEvent, ...] = ()
        self.last_info: ApplyInfo | None = None
        self._resets = getattr(self, "_resets", 0) + 1
        self._pending: list[BookEvent] = []
        self._dropped: list[tuple[str, Decimal, tuple[str, ...]]] = []
        self._item_ts: str | None = None
        self._position: StreamRecord | None = None

    def apply(self, msg_type: str, item: dict, position: StreamRecord | None = None) -> bool | None:
        resets = self._resets
        self._pending = []
        self._dropped = []
        self._item_ts = item.get("timestamp")
        self._position = position
        result = super().apply(msg_type, item)

        events = self._pending
        dropped_ids = {oid for _, _, oids in self._dropped for oid in oids}
        if dropped_ids:
            events = [
                replace(
                    e, qty_after=None, price_after=None, queue_index_after=None, requeued=False,
                    dropped_by_truncation=True,
                )
                if e.order_id in dropped_ids and e.order_id not in self.orders and e.qty_after is not None
                else e
                for e in events
            ]
        if result is True:
            self.sync_status = SyncStatus.SYNCED
        elif result is False:
            self.sync_status = SyncStatus.CHECKSUM_FAILED
        self.last_events = tuple(events)
        self.last_info = ApplyInfo(
            msg_type=msg_type,
            result=result,
            snapshot_loaded=msg_type == "snapshot",
            book_reset=self._resets != resets,
            checksum_failed=result is False,
            ignored_unsynced=result is None,
            n_events=len(events),
            truncated=tuple((name, price, len(oids)) for name, price, oids in self._dropped),
        )
        self._pending = []
        return result

    def _apply_update(self, item: dict) -> None:
        # Same loop and order as L3Book._apply_update; state changes go through the
        # inherited _add/_modify/_delete, and events are derived from state around each one.
        index = 0
        for name in BOOK_SIDES:
            for ev in item.get(name, ()):
                kind = ev["event"]
                self.event_counts[kind] += 1
                if kind not in ("add", "modify", "delete"):
                    continue  # counted by the base logic, never applied
                oid = ev["order_id"]
                before = self._order_state(oid)
                bid_before, ask_before = self.best_bid(), self.best_ask()
                if kind == "add":
                    self._add(name, oid, ev["limit_price"], ev["order_qty"])
                elif kind == "modify":
                    self._modify(name, oid, ev["limit_price"], ev["order_qty"])
                else:
                    self._delete(oid)
                after = self._order_state(oid)
                # mirrors L3Book._modify: only a same (side, price) modify keeps its place
                kept_place = kind == "modify" and before is not None and before[:2] == (name, ev["limit_price"])
                self._pending.append(
                    BookEvent(
                        kind=kind,
                        side=name,
                        order_id=oid,
                        price=ev["limit_price"],
                        qty=ev["order_qty"],
                        qty_before=before[3] if before else None,
                        qty_after=after[3] if after else None,
                        price_before=before[1] if before else None,
                        price_after=after[1] if after else None,
                        queue_index_before=before[2] if before else None,
                        queue_index_after=after[2] if after else None,
                        requeued=kind != "delete" and not kept_place and after is not None,
                        dropped_by_truncation=False,
                        best_bid_before=bid_before,
                        best_ask_before=ask_before,
                        best_bid_after=self.best_bid(),
                        best_ask_after=self.best_ask(),
                        item_timestamp=self._item_ts,
                        event_timestamp=ev.get("timestamp"),
                        event_index=index,
                        position=self._position,
                    )
                )
                index += 1

    def _order_state(self, oid: str) -> tuple[str, Decimal, int | None, Decimal] | None:
        loc = self.orders.get(oid)
        if loc is None:
            return None
        name, price = loc
        level = self.side(name)[price]
        queue_index = None
        if self.track_queue:
            queue_index = next(i for i, other in enumerate(level) if other == oid)
        return name, price, queue_index, level[oid]

    def _drop_level(self, side, level) -> None:
        name = "bids" if side is self.bids else "asks"
        self._dropped.append((name, level[0], tuple(level[1])))
        super()._drop_level(side, level)

    # --- read-only accessors -------------------------------------------------

    def level_orders(self, side: str, price: Decimal) -> tuple[tuple[str, Decimal], ...]:
        """(order_id, qty) in FIFO priority order; a copy."""
        return tuple(self.side(side).get(price, {}).items())

    def levels(self, side: str, n: int | None = None) -> tuple[tuple[Decimal, Decimal, int], ...]:
        out = []
        for price, level in self.side(side).items():
            if n is not None and len(out) >= n:
                break
            out.append((price, sum(level.values(), Decimal(0)), len(level)))
        return tuple(out)

    def level_qty(self, side: str, price: Decimal) -> Decimal:
        return sum(self.side(side).get(price, {}).values(), Decimal(0))

    def view(self) -> ReadOnlyBookView:
        return ReadOnlyBookView(self)
