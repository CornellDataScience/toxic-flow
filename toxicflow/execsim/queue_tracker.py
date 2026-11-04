"""Queue position of one hypothetical order, followed through historical BookEvents.

The tracker holds its own copy of the orders that were ahead of us when we
were inserted (FIFO: every order already at our price, I2) and updates it from
the events SimL3Book emits. It never sees or touches the book.

Two outputs are kept apart on purpose:

    queue advancement   an order ahead of us shrank or left. Trade or cancel,
                        the queue moves the same way; this is never a fill.
    ReachEvent          a reduction that could only have reached us once the
                        queue ahead is empty (§6.3 reach test). It is a
                        candidate; evidence and policies decide later whether
                        it counts as a fill.

The engine must pass only events from items whose checksum passed (R8), in
stream order, and must end the order on any discontinuity (§7.3).
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from decimal import Decimal

from .models import EndReason, Mechanism, ReachEvent, Side
from .sim_book import BookEvent

INCREASE_POLICIES = ("keep", "back")


class QueueTracker:
    def __init__(
        self,
        side: Side,
        price: Decimal,
        ahead: Sequence[tuple[str, Decimal]],
        increase_policy: str = "keep",
    ):
        """`ahead` is SimL3Book.level_orders(side.book_side, price) at insertion, in priority order.

        increase_policy: on a same-price qty increase of an order ahead, `keep`
        leaves it ahead (mirrors L3Book), `back` drops it behind us (Kraken's
        amend rule). OQ-2 is open, so both are offered.
        """
        if increase_policy not in INCREASE_POLICIES:
            raise ValueError(f"increase_policy must be one of {INCREASE_POLICIES}, got {increase_policy!r}")
        self.side = side
        self.price = price
        self.increase_policy = increase_policy
        self.ahead_ids: dict[str, Decimal] = dict(ahead)  # order_id -> last known qty, priority order
        self.initial_ahead_qty = self.ahead_qty
        self.initial_ahead_orders = len(self.ahead_ids)
        self.advanced_qty = Decimal(0)  # total quantity that left the queue ahead of us
        self.front_ref: tuple[int | None, int] | None = None  # event that emptied the queue ahead
        self.front_ts_ns: int | None = None
        self.invalidated: EndReason | None = None  # set once the queue can no longer be followed
        self.diag: Counter = Counter()

    @property
    def ahead_qty(self) -> Decimal:
        return sum(self.ahead_ids.values(), Decimal(0))

    @property
    def ahead_orders(self) -> int:
        return len(self.ahead_ids)

    @property
    def at_front(self) -> bool:
        return not self.ahead_ids

    def invalidate(self, reason: EndReason) -> None:
        """Stop tracking: later events are ignored. The first reason wins."""
        if self.invalidated is None:
            self.invalidated = reason

    # --- event handling ---------------------------------------------------

    def on_event(self, ev: BookEvent, item_ts_ns: int | None = None) -> ReachEvent | None:
        """Apply one historical event (rows 1 to 8 of §6.3). Returns a candidate or None.

        `item_ts_ns` is the engine's guarded item clock (R2, R3); it is copied
        onto the ReachEvent and front_ts_ns and is not otherwise used.
        """
        if self.invalidated is not None:
            self.diag["events_after_invalid"] += 1
            return None
        if ev.order_id in self.ahead_ids:
            self._advance(ev, item_ts_ns)  # row 1: queue advancement, never a fill
            return None
        if ev.side == self.side.book_side:
            if ev.kind in ("add", "modify") and ev.qty_before is None:
                return None  # rows 2 and 8: new order (or unknown-ID modify) goes behind us
            if ev.kind == "add":
                return None  # row 2: re-add of a known ID behind us or elsewhere; it lands behind us
            return self._reach(ev, item_ts_ns)  # rows 3 to 6
        if ev.kind == "add":
            return self._cross_arrival(ev, item_ts_ns)  # row 7
        return None

    def on_events(self, events: Iterable[BookEvent], item_ts_ns: int | None = None) -> list[ReachEvent]:
        """Feed one item's events in order; returns the candidates."""
        out = []
        for ev in events:
            reach = self.on_event(ev, item_ts_ns)
            if reach is not None:
                out.append(reach)
        return out

    def on_truncation(self, truncated: Iterable[tuple[str, Decimal, int]]) -> None:
        """Check ApplyInfo.truncated: depth truncation emits no events for the orders it forgets.

        If a dropped level on our side is at our price or better, our price is
        outside the subscribed depth and its queue can no longer be followed.
        """
        for name, price, _ in truncated:
            if name == self.side.book_side and not self.side.better(self.price, price):
                self.invalidate(EndReason.OUT_OF_DEPTH)

    def _advance(self, ev: BookEvent, item_ts_ns: int | None) -> None:
        oid = ev.order_id
        known = self.ahead_ids[oid]
        if ev.qty_before is not None and ev.qty_before != known:
            self.diag["ahead_qty_mismatch"] += 1  # tracker and book disagree: an I3 violation
        if ev.dropped_by_truncation:
            # an order at our price was forgotten by depth truncation, so our level left the depth
            self._leave(oid)
            self.invalidate(EndReason.OUT_OF_DEPTH)
        elif ev.kind == "delete" or ev.qty_after is None:
            self._leave(oid)  # row 1a
        elif ev.requeued or ev.price_after != self.price or ev.side != self.side.book_side:
            self._leave(oid)  # row 1d: re-added or moved price, priority lost
            self.diag["ahead_requeued"] += 1
        elif ev.qty_after < known:
            self.ahead_ids[oid] = ev.qty_after  # row 1b: keeps priority
            self.advanced_qty += known - ev.qty_after
        elif ev.qty_after > known:
            self.diag["ahead_increase"] += 1  # row 1c
            if self.increase_policy == "back":
                self._leave(oid)
            else:
                self.ahead_ids[oid] = ev.qty_after
        if self.at_front and self.front_ref is None:
            self.front_ref = _ref(ev)
            self.front_ts_ns = item_ts_ns

    def _leave(self, oid: str) -> None:
        self.advanced_qty += self.ahead_ids.pop(oid)

    def _reach(self, ev: BookEvent, item_ts_ns: int | None) -> ReachEvent | None:
        if ev.dropped_by_truncation:
            return None  # forgotten by depth truncation, not removed by the market
        if ev.kind == "modify" and (ev.requeued or ev.price_after != ev.price_before):
            return None  # re-queue at another price, not a fill
        removed = ev.reduction
        if removed <= 0:
            return None  # qty increase, no-op, or delete of an unknown ID
        p = ev.price_before
        if self.side.better(p, self.price):
            return None  # row 5: priced ahead of us
        if p == self.price and self.ahead_ids:
            self.diag["fifo_anomaly"] += 1  # row 4: behind us while someone is still ahead
            return None
        best_before = ev.best_bid_before if ev.side == "bids" else ev.best_ask_before
        if best_before != p:
            return None  # row 6: executions happen only at the touch
        mechanism = Mechanism.LEVEL if p == self.price else Mechanism.INTERCEPT
        return ReachEvent(_ref(ev), item_ts_ns, p, removed, mechanism)

    def _cross_arrival(self, ev: BookEvent, item_ts_ns: int | None) -> ReachEvent | None:
        if ev.dropped_by_truncation or ev.qty_after is None:
            return None
        crosses = ev.price <= self.price if self.side is Side.BUY else ev.price >= self.price
        if not crosses:
            return None
        if self.ahead_ids:
            # a resting order that crosses our price means no historical order sat at it;
            # with a non-empty ahead set the state is inconsistent, so claim nothing
            self.diag["cross_with_ahead"] += 1
            return None
        return ReachEvent(_ref(ev), item_ts_ns, ev.price, ev.qty_after, Mechanism.CROSS_ARRIVAL)

    # --- audit ------------------------------------------------------------

    def audit(self, level: Sequence[tuple[str, Decimal]]) -> list[str]:
        """Invariant I3 against SimL3Book.level_orders(side, price): ahead IDs are at our level
        with the quantities we hold, and in the same relative order. Returns the problems found."""
        book = dict(level)
        problems = []
        for oid, qty in self.ahead_ids.items():
            if oid not in book:
                problems.append(f"{oid} is ahead but not at {self.price}")
            elif book[oid] != qty:
                problems.append(f"{oid} qty {qty} != book {book[oid]}")
        order = [oid for oid in book if oid in self.ahead_ids]
        if order != [oid for oid in self.ahead_ids if oid in book]:
            problems.append("ahead order differs from book priority")
        return problems


def _ref(ev: BookEvent) -> tuple[int | None, int]:
    return (ev.position.seq if ev.position is not None else None, ev.event_index)
