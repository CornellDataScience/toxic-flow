"""Fill simulators for hypothetical passive orders.

QueueAwareSimulator: our order joins the BACK of the queue. It fills only after
every order ahead of it has been traded through or cancelled.
NaiveSimulator: fills as soon as a trade prints at our price. This is the
"standard modeling error" the proposal wants to quantify.

Both are counterfactual: our orders never affect the real market.
"""
from collections import OrderedDict
from dataclasses import dataclass, field

from .book import OrderBook
from .events import BookEvent, Trade


@dataclass
class SimOrder:
    oid: str
    side: str            # "bid" (we buy) | "ask" (we sell)
    price: float
    qty: float
    ts_placed: float
    remaining: float
    ahead: OrderedDict = field(default_factory=OrderedDict)  # order_id -> qty
    live: bool = True

    @property
    def qty_ahead(self):
        return sum(self.ahead.values())


@dataclass(frozen=True)
class Fill:
    oid: str
    ts: float
    side: str
    price: float
    qty: float
    queue_at_placement: float
    time_to_fill: float


def _passive_side(trade: Trade) -> str:
    """Which resting side a trade executed against."""
    return "bid" if trade.aggressor == "sell" else "ask"


def _crosses_through(order: SimOrder, trade: Trade) -> bool:
    """Trade printed at a price worse than ours => our whole level was consumed."""
    return trade.price < order.price if order.side == "bid" else trade.price > order.price


class _BaseSimulator:
    name = "base"

    def __init__(self, book: OrderBook):
        self.book = book
        self.orders: dict[str, SimOrder] = {}   # every order ever placed
        self._live: dict[str, SimOrder] = {}    # only working orders (fast loop)
        self.fills: list[Fill] = []

    def place(self, oid, side, price, qty, ts):
        best_opp = self.book.best_ask() if side == "bid" else self.book.best_bid()
        if best_opp is not None and (
            (side == "bid" and price >= best_opp) or (side == "ask" and price <= best_opp)
        ):
            raise ValueError("passive order would cross the spread")
        order = SimOrder(oid, side, price, qty, ts, remaining=qty)
        self._init_queue(order)
        self.orders[oid] = order
        self._live[oid] = order
        return order

    def cancel(self, oid):
        order = self._live.pop(oid, None)
        if order is not None:
            order.live = False

    def live_orders(self):
        return list(self._live.values())

    def _fill(self, order, ts, qty):
        qty = min(qty, order.remaining)
        if qty <= 0:
            return
        order.remaining -= qty
        if order.remaining <= 1e-12:
            order.remaining, order.live = 0.0, False
            self._live.pop(order.oid, None)
        self.fills.append(Fill(order.oid, ts, order.side, order.price, qty,
                               queue_at_placement=getattr(order, "_q0", 0.0),
                               time_to_fill=ts - order.ts_placed))

    # hooks
    def _init_queue(self, order):
        order._q0 = 0.0

    def on_book_event(self, ev: BookEvent, prev_qty):
        pass

    def on_trade(self, tr: Trade):
        raise NotImplementedError


class NaiveSimulator(_BaseSimulator):
    """Fill everything the moment a trade touches our price."""
    name = "naive"

    def on_trade(self, tr: Trade):
        side = _passive_side(tr)
        for o in self.live_orders():
            if o.side == side and (tr.price == o.price or _crosses_through(o, tr)):
                self._fill(o, tr.ts, o.remaining)


class QueueAwareSimulator(_BaseSimulator):
    """FIFO queue position tracking.

    ASSUMPTION on event ordering: for each match, the Trade arrives BEFORE the
    BookEvent that reduces/deletes the filled resting order. If the real feed
    sends them the other way round, a delete would shrink our queue first and
    the trade would then look like it reached us (phantom fill). The data group
    must confirm the ordering on real data.
    """
    name = "queue_aware"

    def _init_queue(self, order):
        order.ahead = self.book.queue(order.side, order.price)  # everyone already there
        order._q0 = order.qty_ahead

    def on_book_event(self, ev: BookEvent, prev_qty):
        if ev.action == "add":
            return  # new orders join behind us
        for o in self.live_orders():
            if ev.order_id not in o.ahead:
                continue
            if ev.action == "delete" or ev.qty <= 0:
                del o.ahead[ev.order_id]            # cancelled or fully filled
            elif prev_qty is not None and ev.qty > prev_qty:
                del o.ahead[ev.order_id]            # size increase -> moved behind us
            else:
                o.ahead[ev.order_id] = min(o.ahead[ev.order_id], ev.qty)

    def on_trade(self, tr: Trade):
        side = _passive_side(tr)
        for o in self.live_orders():
            if o.side != side:
                continue
            if _crosses_through(o, tr):
                o.ahead.clear()
                self._fill(o, tr.ts, o.remaining)
            elif tr.price == o.price:
                vol = tr.qty
                for oid in list(o.ahead):            # trade eats the queue front-first
                    take = min(o.ahead[oid], vol)
                    o.ahead[oid] -= take
                    vol -= take
                    if o.ahead[oid] <= 1e-12:
                        del o.ahead[oid]
                    if vol <= 1e-12:
                        break
                if vol > 1e-12:                      # volume left over reaches us
                    self._fill(o, tr.ts, vol)
