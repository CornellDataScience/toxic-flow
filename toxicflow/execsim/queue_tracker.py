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

Diagnostic counters (`diag`) never change the queue or candidacy:

    behind_reduction    a reduction behind us at our price while orders are
                        still ahead; ordinarily a cancel
    fifo_anomaly        the subset that is a partial reduction at the touch,
                        which would be an execution out of FIFO order (§6.5.1)
    ahead_increase      same-price qty increase of an order ahead (row 1c); the book
                        re-queues it, so it is also counted in ahead_requeued
    ahead_requeued      an order ahead lost priority: re-added, moved price, or grew (row 1d)
    ahead_qty_mismatch  tracked qty differs from the event's prior qty (I3)
    crossing            opposite-side orders that came to rest at or through our price
                        (new arrivals and re-priced orders; the first one is kept as `cross`)
    cross_with_ahead    crossing arrival while orders are still ahead (no candidate)
    cross_readd         re-add of an order already resting through our price
                        (same liquidity; no second candidate)

`cross` (a CrossEvent) is the first such crossing. Recording it changes neither the queue nor
which events are candidates: what a crossing means for fills is a settlement rule
(execution.CROSS_RULES), applied afterwards to the same logged candidates.
    events_after_invalid  events ignored after invalidation
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from decimal import Decimal

from .models import CrossEvent, EndReason, Mechanism, ReachEvent, Side
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

        increase_policy: what to do with a same-price qty increase of an order ahead that the
        book did NOT re-queue: `keep` leaves it ahead, `back` drops it behind us. OQ-2 is now
        answered by the data: keeping the position broke Kraken's L3 checksum, so L3Book sends
        an increased order to the back and SimL3Book reports it as `requeued`. The tracker always
        follows a requeued event (row 1d), whatever this setting; the setting only matters for
        hand-built events and is kept so existing configurations still load.
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
        self.cross: CrossEvent | None = None  # first opposite-side order resting at or through our price
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
        if ev.kind == "modify":
            self._cross_reprice(ev, item_ts_ns)  # recorded, never a candidate (row 7 is add only)
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
            self._leave(oid)  # row 1d: re-added, moved price or grew; priority lost
            self.diag["ahead_requeued"] += 1
            if ev.price_after == self.price and ev.side == self.side.book_side and ev.qty_after > known:
                self.diag["ahead_increase"] += 1  # row 1c as the book now applies it
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
        best_before = ev.best_bid_before if ev.side == "bids" else ev.best_ask_before
        if p == self.price and self.ahead_ids:
            # row 4: behind us while someone is still ahead. Usually an ordinary cancel; only a
            # partial reduction at the touch (the "usually a fill" case, §6.5.2) contradicts FIFO.
            self.diag["behind_reduction"] += 1
            if ev.kind == "modify" and best_before == p:
                self.diag["fifo_anomaly"] += 1
            return None
        if best_before != p:
            return None  # row 6: executions happen only at the touch
        mechanism = Mechanism.LEVEL if p == self.price else Mechanism.INTERCEPT
        return ReachEvent(_ref(ev), item_ts_ns, p, removed, mechanism)

    def _cross_arrival(self, ev: BookEvent, item_ts_ns: int | None) -> ReachEvent | None:
        if ev.dropped_by_truncation or ev.qty_after is None:
            return None
        if not self._crosses(ev.price):
            return None
        if ev.qty_before is not None and self._crosses(ev.price_before):
            # re-add of an order that was already resting through our price: the same
            # liquidity, already offered to us when it first arrived; never count it twice
            self.diag["cross_readd"] += 1
            return None
        self._note_cross(ev, item_ts_ns)
        if self.ahead_ids:
            # a resting order that crosses our price means no historical order sat at it;
            # with a non-empty ahead set the state is inconsistent, so claim nothing
            self.diag["cross_with_ahead"] += 1
            return None
        return ReachEvent(_ref(ev), item_ts_ns, ev.price, ev.qty_after, Mechanism.CROSS_ARRIVAL)

    def _cross_reprice(self, ev: BookEvent, item_ts_ns: int | None) -> None:
        """An opposite-side modify that leaves an order at or through our price which was not
        there before (a re-price, or the modify of an unknown ID that the book treats as an add)."""
        if ev.dropped_by_truncation or ev.qty_after is None or not self._crosses(ev.price_after):
            return
        if ev.qty_before is not None and self._crosses(ev.price_before):
            return  # it already rested through our price
        self._note_cross(ev, item_ts_ns)

    def _note_cross(self, ev: BookEvent, item_ts_ns: int | None) -> None:
        self.diag["crossing"] += 1
        if self.cross is None:
            self.cross = CrossEvent(_ref(ev), item_ts_ns, ev.price_after, ev.qty_after, self.ahead_qty, ev.kind)

    def _crosses(self, opposite_price: Decimal) -> bool:
        """An opposite-side price at or through ours (ask <= our bid, bid >= our ask)."""
        return opposite_price <= self.price if self.side is Side.BUY else opposite_price >= self.price

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
