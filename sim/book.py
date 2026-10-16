"""L3 order book: every resting order, in FIFO (time-priority) order per level."""
from collections import OrderedDict

from .events import BookEvent


class OrderBook:
    def __init__(self):
        # side -> {price -> OrderedDict(order_id -> qty)}; dict order = queue order
        self.levels = {"bid": {}, "ask": {}}
        self.orders = {}          # order_id -> (side, price)
        self.unknown_events = 0   # updates for orders we never saw (desync signal)

    def apply(self, ev: BookEvent):
        """Apply one event. Returns the order's previous qty (None for adds/unknown)."""
        if ev.action == "add":
            level = self.levels[ev.side].setdefault(ev.price, OrderedDict())
            level[ev.order_id] = ev.qty
            self.orders[ev.order_id] = (ev.side, ev.price)
            return None

        if ev.order_id not in self.orders:
            self.unknown_events += 1
            return None
        side, price = self.orders[ev.order_id]
        level = self.levels[side][price]
        prev = level[ev.order_id]

        if ev.action == "delete" or (ev.action == "modify" and ev.qty <= 0):
            del level[ev.order_id]
            del self.orders[ev.order_id]
            if not level:
                del self.levels[side][price]
        elif ev.action == "modify":
            if ev.qty > prev:
                # ASSUMPTION: a size increase loses time priority (back of queue).
                # A decrease (partial fill or partial cancel) keeps its place.
                # Verify against the exchange's matching rules.
                del level[ev.order_id]
            level[ev.order_id] = ev.qty
        else:
            raise ValueError(f"unknown action {ev.action!r}")
        return prev

    # --- queries -------------------------------------------------------
    def best_bid(self):
        return max(self.levels["bid"]) if self.levels["bid"] else None

    def best_ask(self):
        return min(self.levels["ask"]) if self.levels["ask"] else None

    def mid(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else (b + a) / 2

    def spread(self):
        b, a = self.best_bid(), self.best_ask()
        return None if b is None or a is None else a - b

    def level_qty(self, side, price):
        return sum(self.levels[side].get(price, {}).values())

    def queue(self, side, price):
        """Copy of the FIFO queue at a level: OrderedDict(order_id -> qty)."""
        return OrderedDict(self.levels[side].get(price, {}))
