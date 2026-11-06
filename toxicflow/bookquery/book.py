"""Order-by-order (L3) book state in exact integer units.

Each price level is a FIFO queue of resting orders: a dict order_id -> lots,
in time priority (front of the queue first). The priority rules below were
checked against Kraken's L3 checksums on recorded data:

- add / snapshot: the order joins the back of its level.
- modify, size decreased (partial fill or amend down): keeps its position.
- modify, size increased: moves to the back of its level. Exchanges do this so
  nobody can jump the queue by enlarging an order that is already near the front.
- modify at a different price (or for an unknown order): treated as a new
  order at the back of the new level. Kraken usually sends price amends as
  delete + add, which has the same effect.
- delete: removes the order. This is the only way an order leaves the queue;
  fills shrink it through modify and the delete follows separately.

Kraken only streams the top `depth` levels per side. When a better level
appears and pushes the worst level past `depth`, Kraken stops updating it
without sending deletes, so the book drops it after each message (truncate).
"""

from collections.abc import Hashable
from dataclasses import dataclass
from decimal import Decimal
from itertools import islice

from sortedcontainers import SortedDict

from toxicflow.kraken.book import PairInfo
from toxicflow.kraken.checksum import crc32

from .units import decimal_to_ticks, ticks_to_decimal

BID, ASK = 0, 1
SNAPSHOT, ADD, MODIFY, DELETE = 0, 1, 2, 3

SIDE_CODES = {"bid": BID, "bids": BID, "buy": BID, "b": BID, "ask": ASK, "asks": ASK, "sell": ASK, "s": ASK, "a": ASK}
EVENT_CODES = {"snapshot": SNAPSHOT, "add": ADD, "modify": MODIFY, "delete": DELETE}
SIDE_NAMES = ("bid", "ask")

CHECKSUM_LEVELS = 10

PriceLike = Decimal | str | float | int
Level = dict[Hashable, int]  # order_id -> lots, front of queue first


def side_code(side: str | int) -> int:
    if isinstance(side, int):
        if side not in (BID, ASK):
            raise ValueError(f"bad side {side!r}")
        return side
    try:
        return SIDE_CODES[side.lower()]
    except KeyError:
        raise ValueError(f"bad side {side!r}; use 'bid'/'ask' (or 'buy'/'sell')") from None


def _neg(price: int) -> int:
    return -price


@dataclass(frozen=True)
class Best:
    """Top of book. Prices/sizes are None when that side is empty."""

    bid_price: Decimal | None
    bid_qty: Decimal | None
    ask_price: Decimal | None
    ask_qty: Decimal | None

    @property
    def mid(self) -> Decimal | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return (self.bid_price + self.ask_price) / 2

    @property
    def spread(self) -> Decimal | None:
        if self.bid_price is None or self.ask_price is None:
            return None
        return self.ask_price - self.bid_price


class Book:
    """L3 book. `bids`/`asks` map price ticks -> Level; index 0 is the best price.

    Prices and sizes are integer ticks/lots internally. The public accessors
    (best, depth, queue, queue_ahead) take and return Decimal.
    `ts` is the timestamp of the last event applied (None for a fresh book).
    """

    __slots__ = ("pair", "depth", "bids", "asks", "orders", "ts")

    def __init__(self, pair: PairInfo, depth: int | None = None):
        self.pair = pair
        self.depth = depth
        self.bids: SortedDict = SortedDict(_neg)
        self.asks: SortedDict = SortedDict()
        self.orders: dict[Hashable, tuple[int, int]] = {}  # order_id -> (side, price ticks)
        self.ts: int | None = None

    # -- mutation (integer units) -----------------------------------------

    def apply(self, kind: int, side: int, oid: Hashable, price: int, qty: int) -> None:
        if kind == DELETE:
            self._delete(oid)
        elif kind == MODIFY:
            loc = self.orders.get(oid)
            if loc == (side, price):
                level = self._side(side)[price]
                if qty > level[oid]:
                    del level[oid]  # re-inserting moves it to the back of the dict
                level[oid] = qty
            else:
                self._add(side, oid, price, qty)
        else:  # ADD / SNAPSHOT
            self._add(side, oid, price, qty)

    def truncate(self) -> None:
        """Drop levels beyond `depth` (call once per exchange message)."""
        if self.depth is None:
            return
        for book_side in (self.bids, self.asks):
            while len(book_side) > self.depth:
                _, level = book_side.popitem(-1)
                for oid in level:
                    del self.orders[oid]

    def _side(self, side: int) -> SortedDict:
        return self.bids if side == BID else self.asks

    def _add(self, side: int, oid: Hashable, price: int, qty: int) -> None:
        if oid in self.orders:
            self._delete(oid)
        book_side = self._side(side)
        level = book_side.get(price)
        if level is None:
            level = book_side[price] = {}
        level[oid] = qty
        self.orders[oid] = (side, price)

    def _delete(self, oid: Hashable) -> None:
        loc = self.orders.pop(oid, None)
        if loc is None:
            return  # e.g. an order that was truncated out of the visible depth
        book_side = self._side(loc[0])
        level = book_side[loc[1]]
        del level[oid]
        if not level:
            del book_side[loc[1]]

    def copy(self) -> "Book":
        new = Book(self.pair, self.depth)
        new.bids = SortedDict(_neg, {p: dict(lvl) for p, lvl in self.bids.items()})
        new.asks = SortedDict({p: dict(lvl) for p, lvl in self.asks.items()})
        new.orders = dict(self.orders)
        new.ts = self.ts
        return new

    def checksum(self) -> int:
        """Kraken L3 CRC32: top 10 levels, asks then bids, one price+qty per order.

        With integer ticks/lots, Kraken's "format at precision, drop the
        decimal point and leading zeros" is just str() of the integer.
        """
        parts: list[str] = []
        for book_side in (self.asks, self.bids):
            for price, level in islice(book_side.items(), CHECKSUM_LEVELS):
                p = str(price)
                parts.extend(p + str(q) for q in level.values())
        return crc32("".join(parts))

    # -- integer-unit reads ---------------------------------------------------

    def best_ticks(self) -> tuple[int, int, int, int]:
        """(bid ticks, bid lots, ask ticks, ask lots); -1 for an empty side."""
        if self.bids:
            bp, bl = self.bids.peekitem(0)
            bq = sum(bl.values())
        else:
            bp = bq = -1
        if self.asks:
            ap, al = self.asks.peekitem(0)
            aq = sum(al.values())
        else:
            ap = aq = -1
        return bp, bq, ap, aq

    def at_depth_edge(self, side: int, price: int) -> bool:
        """True if `price` is beyond what the feed shows for this side.

        Only possible when the side holds `depth` levels: with fewer, the feed
        is showing the whole side and an absent price really is empty.
        """
        book_side = self._side(side)
        if self.depth is None or len(book_side) < self.depth:
            return False
        worst = book_side.peekitem(-1)[0]
        return price > worst if side == ASK else price < worst

    # -- Decimal accessors ------------------------------------------------------

    def px(self, ticks: int) -> Decimal:
        return ticks_to_decimal(ticks, self.pair.price_precision)

    def qty(self, lots: int) -> Decimal:
        return ticks_to_decimal(lots, self.pair.qty_precision)

    def price_ticks(self, price: PriceLike) -> int:
        return decimal_to_ticks(price, self.pair.price_precision)

    def best(self) -> Best:
        bp, bq, ap, aq = self.best_ticks()
        return Best(
            self.px(bp) if bp >= 0 else None,
            self.qty(bq) if bp >= 0 else None,
            self.px(ap) if ap >= 0 else None,
            self.qty(aq) if ap >= 0 else None,
        )

    def depth_levels(self, levels: int = 10) -> dict[str, list[tuple[Decimal, Decimal]]]:
        """Aggregated (price, total size) per level, best first, both sides."""
        return {
            name: [(self.px(p), self.qty(sum(lvl.values()))) for p, lvl in islice(book_side.items(), levels)]
            for name, book_side in (("bids", self.bids), ("asks", self.asks))
        }

    def queue(self, side: str | int, price: PriceLike) -> list[tuple[Hashable, Decimal]]:
        """Orders resting at one level, front of the queue first. [] if none."""
        level = self._side(side_code(side)).get(self.price_ticks(price), {})
        return [(oid, self.qty(q)) for oid, q in level.items()]

    def queue_ahead(self, side: str | int, price: PriceLike, order_id: Hashable | None = None) -> Decimal:
        """Size queued ahead at this level only (better prices are not counted).

        With order_id: total size of orders in front of it (KeyError if it
        isn't resting at this level). Without: the whole level, i.e. what a new
        order joining the back of the queue right now would wait behind.
        """
        s, p = side_code(side), self.price_ticks(price)
        level = self._side(s).get(p, {})
        if order_id is None:
            return self.qty(sum(level.values()))
        if order_id not in level:
            raise KeyError(f"order {order_id!r} is not resting at {SIDE_NAMES[s]} {price}")
        ahead = 0
        for oid, q in level.items():
            if oid == order_id:
                break
            ahead += q
        return self.qty(ahead)
