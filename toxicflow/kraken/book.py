"""Local order book reconstruction from Kraken WebSocket v2 messages.

L2Book: aggregated price levels from the `book` channel.
L3Book: individual orders from the `level3` channel.

Both books follow the same protocol: feed every snapshot/update item to
`apply()`, which returns True if the post-update checksum matches Kraken's,
False on mismatch (book is then marked unsynced and must be resubscribed), or
None if the update was ignored because no snapshot has arrived yet.
"""

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from itertools import islice

from sortedcontainers import SortedDict

from .checksum import crc32, fmt

CHECKSUM_LEVELS = 10


@dataclass(frozen=True)
class PairInfo:
    symbol: str
    price_precision: int
    qty_precision: int


def _neg(price: Decimal) -> Decimal:
    return -price


class _BaseBook:
    def __init__(self, pair: PairInfo, depth: int):
        self.pair = pair
        self.symbol = pair.symbol
        self.depth = depth
        self.reset()

    def reset(self) -> None:
        # bids sorted high->low, asks low->high: index 0 is always best.
        self.bids = SortedDict(_neg)
        self.asks = SortedDict()
        self.synced = False
        self.timestamp: str | None = None

    def side(self, name: str) -> SortedDict:
        return self.bids if name == "bids" else self.asks

    def apply(self, msg_type: str, item: dict) -> bool | None:
        if msg_type == "snapshot":
            self.reset()
            self.synced = True
            self._load_snapshot(item)
        elif not self.synced:
            return None
        else:
            self._apply_update(item)
        for side in (self.bids, self.asks):
            while len(side) > self.depth:
                self._drop_level(side, side.popitem(-1))
        self.timestamp = item.get("timestamp", self.timestamp)
        ok = self.checksum() == item["checksum"]
        if not ok:
            self.synced = False
        return ok

    def checksum(self) -> int:
        return crc32(self._checksum_str(self.asks) + self._checksum_str(self.bids))

    def best_bid(self) -> Decimal | None:
        return self.bids.peekitem(0)[0] if self.bids else None

    def best_ask(self) -> Decimal | None:
        return self.asks.peekitem(0)[0] if self.asks else None

    def mid(self) -> Decimal | None:
        bid, ask = self.best_bid(), self.best_ask()
        return (bid + ask) / 2 if bid is not None and ask is not None else None

    def _fmt_pair(self, price: Decimal, qty: Decimal) -> str:
        return fmt(price, self.pair.price_precision) + fmt(qty, self.pair.qty_precision)

    def _load_snapshot(self, item: dict) -> None: ...
    def _apply_update(self, item: dict) -> None: ...
    def _drop_level(self, side: SortedDict, level: tuple) -> None: ...
    def _checksum_str(self, side: SortedDict) -> str: ...


class L2Book(_BaseBook):
    """price -> aggregate qty per side."""

    def _load_snapshot(self, item: dict) -> None:
        self._apply_update(item)

    def _apply_update(self, item: dict) -> None:
        for name in ("bids", "asks"):
            side = self.side(name)
            for lvl in item.get(name, ()):
                if lvl["qty"] == 0:
                    side.pop(lvl["price"], None)
                else:
                    side[lvl["price"]] = lvl["qty"]

    def _drop_level(self, side, level) -> None:
        pass

    def _checksum_str(self, side) -> str:
        return "".join(self._fmt_pair(p, q) for p, q in islice(side.items(), CHECKSUM_LEVELS))

    def level_qty(self, name: str, price: Decimal) -> Decimal:
        return self.side(name).get(price, Decimal(0))


class L3Book(_BaseBook):
    """price -> {order_id: qty} per side.

    Orders within a level are kept in the order Kraken lists them (dict
    insertion order), because the L3 checksum hashes them in that order.
    """

    def reset(self) -> None:
        super().reset()
        self.orders: dict[str, tuple[str, Decimal]] = {}  # order_id -> (side, price)
        self.event_counts: Counter = Counter()

    def _load_snapshot(self, item: dict) -> None:
        for name in ("bids", "asks"):
            for o in item.get(name, ()):
                self._add(name, o["order_id"], o["limit_price"], o["order_qty"])

    def _apply_update(self, item: dict) -> None:
        for name in ("bids", "asks"):
            for ev in item.get(name, ()):
                kind = ev["event"]
                self.event_counts[kind] += 1
                if kind == "add":
                    self._add(name, ev["order_id"], ev["limit_price"], ev["order_qty"])
                elif kind == "modify":
                    self._modify(name, ev["order_id"], ev["limit_price"], ev["order_qty"])
                elif kind == "delete":
                    self._delete(ev["order_id"])

    def _add(self, name: str, oid: str, price: Decimal, qty: Decimal) -> None:
        if oid in self.orders:
            self._delete(oid)
        self.side(name).setdefault(price, {})[oid] = qty
        self.orders[oid] = (name, price)

    def _modify(self, name: str, oid: str, price: Decimal, qty: Decimal) -> None:
        if self.orders.get(oid) == (name, price):
            self.side(name)[price][oid] = qty  # qty change keeps the order's position
        else:
            self._add(name, oid, price, qty)  # unknown order or price moved: append to level

    def _delete(self, oid: str) -> None:
        loc = self.orders.pop(oid, None)
        if loc is None:
            return
        name, price = loc
        side = self.side(name)
        level = side[price]
        del level[oid]
        if not level:
            del side[price]

    def _drop_level(self, side, level) -> None:
        for oid in level[1]:
            self.orders.pop(oid, None)

    def _checksum_str(self, side) -> str:
        return "".join(
            self._fmt_pair(price, qty)
            for price, level in islice(side.items(), CHECKSUM_LEVELS)
            for qty in level.values()
        )
