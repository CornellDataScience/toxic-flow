"""Event log: every L3 order event and every trade, linked to each other.

One pass over the L3 recordings (through IntervalTracker, so validity follows
the same rules as scripts/valid_intervals.py) turns each level3 message into
one row per order event, with the book state around it. Trades from the public
stream are loaded first and matched to L3 reductions, so each reduction gets a
label and each trade lists the L3 events it explains.

Labels (column `action`):
  snapshot      order listed in a snapshot (rows of one snapshot share msg_id;
                a consumer resets its book when a new snapshot msg_id starts)
  new           add of a fresh order
  reentry       add of an old order: a level came back into the subscribed depth
                (Kraken re-sends its orders as adds, with their original
                timestamps). Not a market action.
  refill        the order got more visible size without being new: a modify that
                raises qty (book.py sends it to the back of the queue), or an add
                of an order ID that was deleted shortly before
  fill          delete at the touch, fully covered by trade prints
  partial_fill  modify down at the touch, fully covered by trade prints
  cancel        delete / modify down that cannot be a fill (not at the touch), or
                at the touch with a healthy trade feed and no matching trade
  amend         modify that moves the order to another price
  unknown       reduction at the touch while the trade feed has a hole, trades
                cover only part of it, unknown order ID, unchanged modify, or the
                book was not synced / failed its checksum

`evidence` says which rule fired. Rows with valid=0 come from items outside a
valid interval; their state is not trusted and they never consume trades.

Matching (from the queue design doc, sec. 6.5): a trade can explain a reduction
when it hit the same side (taker buy -> asks, taker sell -> bids), at the same
price, within +-match_window of the event's own timestamp. Closest trades are
used first and each trade's quantity is consumed once. Only reductions of an
order that was at the best price just before the event are candidates.
"""

import bisect
import gzip
import heapq
import json
import re
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from .book import L3Book
from .intervals import IntervalTracker
from .ws import loads

S = 1_000_000_000
ZERO = Decimal(0)
SIDE = {"bids": "bid", "asks": "ask"}
MAKER_SIDE = {"buy": "asks", "sell": "bids"}  # taker side -> book side it trades against

EVENT_COLUMNS = [
    "event_id", "msg_id", "symbol", "recv_ns", "item_ts_ns", "event_ts_ns", "side", "event_type",
    "order_id", "price", "qty", "prev_price", "prev_qty", "qty_delta", "queue_pos", "qty_ahead",
    "level_rank", "best_bid", "best_ask", "action", "evidence", "matched_qty", "trade_ids",
    "match_dt_ns", "valid",
]
TRADE_COLUMNS = [
    "trade_id", "symbol", "recv_ns", "ts_ns", "taker_side", "maker_side", "price", "qty",
    "ord_type", "matched_qty", "event_ids", "l3_valid",
]


# -- helpers -----------------------------------------------------------------

@lru_cache(maxsize=8192)
def _epoch_s(base: str) -> int:
    return int(datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())


def parse_ts_ns(ts) -> int | None:
    """'2026-10-18T02:06:55.103101894Z' -> integer ns since the epoch (no float rounding)."""
    if ts is None:
        return None
    if isinstance(ts, int):
        return ts
    base, _, frac = ts.rstrip("Z").partition(".")
    return _epoch_s(base) * S + int(frac.ljust(9, "0")[:9])


def dec(x) -> str:
    """Decimal -> plain string (never 1E-8), '' for None."""
    return "" if x is None else format(x, "f")


def opt(x) -> str:
    return "" if x is None else str(x)


# -- book with per-event state -----------------------------------------------

@dataclass(slots=True)
class BookEvent:
    side: str  # "bids" / "asks"
    kind: str  # snapshot / add / modify / delete
    order_id: str
    price: Decimal | None
    qty: Decimal | None
    event_ts: str | None
    prev_price: Decimal | None = None  # order state just before the event (None: order unknown)
    prev_qty: Decimal | None = None
    at_touch: bool = False  # order was at the best price of its side just before the event
    best_bid: Decimal | None = None  # just before the event
    best_ask: Decimal | None = None
    queue_pos: int | None = None  # 0 = front. After the event for add/modify, before it for delete.
    qty_ahead: Decimal | None = None
    level_rank: int | None = None  # 0 = best price level


class EventL3Book(L3Book):
    """L3Book that records each event with the order's state around it.

    Events are applied one at a time through L3Book's own update logic, so the
    book evolves exactly as L3Book does (same queue rules, same checksums).
    After apply(): `events` holds this item's events, `last_result` the result.
    """

    def reset(self) -> None:
        super().reset()
        self.events: list[BookEvent] = []
        self.last_result: bool | None = None

    def apply(self, msg_type: str, item: dict) -> bool | None:
        self.events = []
        self.last_result = super().apply(msg_type, item)
        return self.last_result

    def _load_snapshot(self, item: dict) -> None:
        for name in ("bids", "asks"):
            for o in item.get(name, ()):
                self._add(name, o["order_id"], o["limit_price"], o["order_qty"])
                e = BookEvent(name, "snapshot", o["order_id"], o["limit_price"], o["order_qty"], o.get("timestamp"))
                e.queue_pos, e.qty_ahead, e.level_rank = self._position(name, o["limit_price"], o["order_id"])
                self.events.append(e)

    def _apply_update(self, item: dict) -> None:
        for name in ("bids", "asks"):
            for ev in item.get(name, ()):
                e = self._before(name, ev)
                super()._apply_update({name: [ev]})  # one event through L3Book's own logic
                if e.kind != "delete":
                    loc = self.orders.get(e.order_id)
                    if loc is not None:
                        e.queue_pos, e.qty_ahead, e.level_rank = self._position(*loc, e.order_id)
                self.events.append(e)

    def _before(self, name: str, ev: dict) -> BookEvent:
        oid = ev["order_id"]
        e = BookEvent(name, ev["event"], oid, ev.get("limit_price"), ev.get("order_qty"), ev.get("timestamp"),
                      best_bid=self.best_bid(), best_ask=self.best_ask())
        loc = self.orders.get(oid)
        if loc is not None:
            side_name, price = loc
            side = self.side(side_name)
            e.prev_price, e.prev_qty = price, side[price][oid]
            e.at_touch = side_name == name and side.peekitem(0)[0] == price
            if e.kind == "delete":
                e.queue_pos, e.qty_ahead, e.level_rank = self._position(side_name, price, oid)
        return e

    def _position(self, name: str, price: Decimal, oid: str) -> tuple[int, Decimal, int]:
        side = self.side(name)
        pos, ahead = 0, ZERO
        for o, q in side[price].items():
            if o == oid:
                break
            pos += 1
            ahead += q
        return pos, ahead, side.index(price)


# -- trades ------------------------------------------------------------------

@dataclass(slots=True)
class Trade:
    trade_id: int
    symbol: str
    recv_ns: int
    ts_ns: int
    taker_side: str
    price: Decimal
    qty: Decimal
    ord_type: str
    matched: Decimal = ZERO
    event_ids: list = field(default_factory=list)

    @property
    def remaining(self) -> Decimal:
        return self.qty - self.matched


_RECV = re.compile(r'^\{"recv_ns":\s*(\d+)')


def read_public(paths: Iterable[Path]) -> Iterator[tuple[int, dict | None]]:
    """Yield (recv_ns, trade message or None) for every line of public recordings.

    Only lines that mention "trade" are JSON-parsed; the L2 book lines are skipped
    cheaply but still give a receive time, which is what feed coverage needs.
    """
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            try:
                for line in fh:
                    m = _RECV.match(line)
                    if not m:
                        continue
                    msg = None
                    if '"trade"' in line:
                        rec = loads(line)
                        msg = rec.get("msg")
                        if not msg or msg.get("channel") != "trade":
                            msg = None
                    yield int(m.group(1)), msg
            except (EOFError, json.JSONDecodeError, UnicodeDecodeError):
                continue  # truncated tail: keep what was read, go to the next file


class TradeIndex:
    """All trades, indexed by (symbol, book side hit, price), plus trade-feed coverage.

    The feed counts as healthy at time t when the public stream was recording
    (no receive gap > gap_s) and trade_id has no jump around t. Holes are padded
    by `pad_ns` because receive time and exchange time differ slightly.
    """

    def __init__(self, records: Iterable[tuple[int, dict | None]], gap_s: float = 5.0, pad_ns: int = S):
        seen: dict[tuple[str, int], Trade] = {}
        holes: list[tuple[int, int]] = []
        self.first_ns = self.last_ns = None
        prev = None
        for recv_ns, msg in records:
            if prev is not None and recv_ns - prev > gap_s * S:
                holes.append((prev, recv_ns))
            prev = recv_ns
            self.first_ns = recv_ns if self.first_ns is None else self.first_ns
            self.last_ns = recv_ns
            if msg is None or msg.get("type") not in ("update", "snapshot"):
                continue
            for t in msg.get("data", ()):
                key = (t["symbol"], int(t["trade_id"]))
                if key not in seen:
                    seen[key] = Trade(int(t["trade_id"]), t["symbol"], recv_ns, parse_ts_ns(t["timestamp"]),
                                      t["side"], t["price"], t["qty"], t.get("ord_type", ""))
        self.trades = sorted(seen.values(), key=lambda t: (t.symbol, t.ts_ns, t.trade_id))

        by_sym: dict[str, list[Trade]] = {}
        for t in self.trades:
            by_sym.setdefault(t.symbol, []).append(t)
        self.id_jumps = 0
        self._holes: dict[str | None, list[tuple[int, int]]] = {None: holes}
        for sym, ts in by_sym.items():
            h = []
            by_id = sorted(ts, key=lambda t: t.trade_id)
            for a, b in zip(by_id, by_id[1:]):
                if b.trade_id - a.trade_id > 1:
                    self.id_jumps += 1
                    h.append((min(a.ts_ns, b.ts_ns), max(a.ts_ns, b.ts_ns)))
            self._holes[sym] = h
        self.recv_holes = len(holes)
        self._merged = {sym: self._merge(self._holes[None] + h, pad_ns) for sym, h in self._holes.items() if sym}
        self._merged_default = self._merge(holes, pad_ns)
        self.pad_ns = pad_ns

        self._index: dict[tuple, tuple[list[int], list[Trade]]] = {}
        for t in self.trades:  # already in ts order within a symbol
            k = (t.symbol, MAKER_SIDE[t.taker_side], t.price)
            ts_list, tr = self._index.setdefault(k, ([], []))
            ts_list.append(t.ts_ns)
            tr.append(t)

    @staticmethod
    def _merge(holes, pad):
        out = []
        for a, b in sorted((a - pad, b + pad) for a, b in holes):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return [tuple(x) for x in out]

    @classmethod
    def from_files(cls, paths: Iterable[Path], gap_s: float = 5.0, pad_ns: int = S) -> "TradeIndex":
        return cls(read_public(paths), gap_s, pad_ns)

    def healthy(self, symbol: str, ts_ns: int) -> bool:
        if self.first_ns is None or not (self.first_ns + self.pad_ns <= ts_ns <= self.last_ns - self.pad_ns):
            return False
        holes = self._merged.get(symbol, self._merged_default)
        i = bisect.bisect_right(holes, (ts_ns, float("inf"))) - 1
        return not (i >= 0 and holes[i][0] <= ts_ns <= holes[i][1])

    def candidates(self, symbol: str, side: str, price: Decimal, ts_ns: int, window_ns: int) -> list[Trade]:
        """Unconsumed trades that hit `side` at `price` within the window, closest first."""
        ts_list, tr = self._index.get((symbol, side, price), ((), ()))
        lo = bisect.bisect_left(ts_list, ts_ns - window_ns)
        hi = bisect.bisect_right(ts_list, ts_ns + window_ns)
        found = [t for t in tr[lo:hi] if t.remaining > 0]
        found.sort(key=lambda t: (abs(t.ts_ns - ts_ns), t.trade_id))
        return found


# -- builder -----------------------------------------------------------------

@dataclass(slots=True, eq=False)
class _Pending:
    """A reduction at the touch waiting for its trade match to be decided."""
    row: list
    sym: str
    side: str
    price: Decimal
    anchor: int
    removed: Decimal
    fill: str
    done: bool = False


class EventLogBuilder:
    """Feed L3 records in order; each event row goes to `emit` (a list in EVENT_COLUMNS order).

    Rows are emitted in stream order. A reduction at the touch is held until the
    exchange clock is 2 x match_window past it, so every event that could compete
    for the same trades has been seen; a trade then goes to the event closest to it
    in time, not to whichever event happened to come first.
    """

    def __init__(self, trades: TradeIndex, emit: Callable[[list], None], gap_s: float = 5.0,
                 match_window_ms: float = 100, reentry_age_ms: float = 100, refill_window_s: float = 60):
        self.trades = trades
        self.emit = emit
        self.window_ns = int(match_window_ms * 1e6)
        self.reentry_ns = int(reentry_age_ms * 1e6)
        self.refill_ns = int(refill_window_s * S)
        self.tracker = IntervalTracker(gap_s, book_factory=EventL3Book, on_item=self._on_item)
        self.msg_id = -1
        self.event_id = 0
        self._deleted: dict[str, OrderedDict] = {}  # symbol -> order_id -> ts of its delete
        self.match_dts: dict[str, list[int]] = {}  # symbol -> trade ts - event ts of each matched trade
        self._rows: deque = deque()  # rows in stream order: list (final) or _Pending
        self._due: list = []  # heap of (anchor + 2 * window, event_id, _Pending)
        self._waiting: dict[tuple, list[_Pending]] = {}  # (sym, side, price) -> unresolved reductions
        self._clock: int | None = None

    def feed(self, recv_ns: int, kind: str, payload: dict) -> None:
        self.msg_id += 1
        self.tracker.feed(recv_ns, kind, payload)

    def finish(self) -> None:
        self.tracker.finish()
        self._resolve(None)

    def _on_item(self, recv_ns: int, msg_type: str, item: dict, book: EventL3Book, valid: bool) -> None:
        sym = item["symbol"]
        item_ts = parse_ts_ns(item.get("timestamp"))
        if book.last_result is None:  # update before any snapshot: the book could not apply it
            for name in ("bids", "asks"):
                for ev in item.get(name, ()):
                    e = BookEvent(name, ev["event"], ev["order_id"], ev.get("limit_price"), ev.get("order_qty"),
                                  ev.get("timestamp"))
                    self._write(sym, recv_ns, item_ts, e, ("unknown", "unsynced"), False)
        else:
            for e in book.events:
                self._write(sym, recv_ns, item_ts, e, self._classify(sym, item_ts, e, valid), valid)
        if item_ts is not None:
            self._clock = item_ts if self._clock is None else max(self._clock, item_ts)
            self._resolve(self._clock)

    def _classify(self, sym, item_ts, e: BookEvent, valid: bool):
        """-> (action, evidence), or a _Pending (row filled in later) for a reduction at the touch."""
        if e.kind == "snapshot":
            return "snapshot", "snapshot"
        deleted = self._deleted.setdefault(sym, OrderedDict())
        if item_ts is not None:
            while deleted and next(iter(deleted.values())) < item_ts - self.refill_ns:
                deleted.popitem(last=False)
        event_ts = parse_ts_ns(e.event_ts)

        if e.kind == "add":
            if e.prev_price is not None or e.order_id in deleted:
                return "refill", "readd"
            if event_ts is not None and item_ts is not None and item_ts - event_ts >= self.reentry_ns:
                return "reentry", "old_timestamp"
            return "new", "fresh"

        if e.kind == "delete":
            deleted[e.order_id] = item_ts if item_ts is not None else 0
        if e.prev_qty is None:
            return "unknown", "unknown_order"
        if e.kind == "modify":
            if e.price != e.prev_price:
                return "amend", "price_change"
            if e.qty > e.prev_qty:
                return "refill", "qty_up"
            if e.qty == e.prev_qty:
                return "unknown", "qty_unchanged"
            removed, fill = e.prev_qty - e.qty, "partial_fill"
        else:
            removed, fill = e.prev_qty, "fill"

        if not e.at_touch:
            return "cancel", "not_at_touch"
        if not valid:
            return "unknown", "book_invalid"
        anchor = event_ts if event_ts is not None else item_ts
        if anchor is None:
            return "unknown", "no_timestamp"
        return _Pending([], sym, e.side, e.prev_price, anchor, removed, fill)

    def _match(self, p: _Pending) -> None:
        """Decide one held reduction. A trade is skipped if another unresolved reduction is closer to it."""
        rivals = [q for q in self._waiting.get((p.sym, p.side, p.price), ()) if q is not p]
        cands = [t for t in self.trades.candidates(p.sym, p.side, p.price, p.anchor, self.window_ns)
                 if not any(abs(t.ts_ns - q.anchor) < abs(t.ts_ns - p.anchor) for q in rivals)]
        avail = sum((t.remaining for t in cands), ZERO)
        r = p.row
        if avail >= p.removed:
            used, need = [], p.removed
            for t in cands:
                take = min(need, t.remaining)
                t.matched += take
                t.event_ids.append(r[0])
                used.append(t)
                self.match_dts.setdefault(p.sym, []).append(t.ts_ns - p.anchor)
                need -= take
                if need == 0:
                    break
            r[19:24] = [p.fill, "trade_match", dec(p.removed), ";".join(str(t.trade_id) for t in used),
                        str(used[0].ts_ns - p.anchor)]
        elif avail > 0:
            r[19:21] = ["unknown", "trades_partial"]
        elif self.trades.healthy(p.sym, p.anchor):
            r[19:21] = ["cancel", "no_trade"]
        else:
            r[19:21] = ["unknown", "no_trade_coverage"]
        p.done = True

    def _resolve(self, clock: int | None) -> None:
        """Decide every held reduction whose window has closed (all of them when clock is None), then flush rows."""
        while self._due and (clock is None or self._due[0][0] < clock):
            _, _, p = heapq.heappop(self._due)
            self._match(p)
            self._waiting[(p.sym, p.side, p.price)].remove(p)
        while self._rows and (isinstance(self._rows[0], list) or self._rows[0].done):
            r = self._rows.popleft()
            self.emit(r if isinstance(r, list) else r.row)

    def _write(self, sym, recv_ns, item_ts, e: BookEvent, cls, valid: bool) -> None:
        if e.kind == "add" or e.kind == "snapshot":
            delta = e.qty
        elif e.kind == "delete":
            delta = None if e.prev_qty is None else -e.prev_qty
        elif e.prev_qty is None or e.price != e.prev_price:
            delta = e.qty
        else:
            delta = e.qty - e.prev_qty
        action, evidence = ("", "") if isinstance(cls, _Pending) else cls
        row = [
            self.event_id, self.msg_id, sym, recv_ns, opt(item_ts), opt(parse_ts_ns(e.event_ts)), SIDE[e.side],
            e.kind, e.order_id, dec(e.price), dec(e.qty), dec(e.prev_price), dec(e.prev_qty), dec(delta),
            opt(e.queue_pos), dec(e.qty_ahead), opt(e.level_rank), dec(e.best_bid), dec(e.best_ask),
            action, evidence, "", "", "", int(valid),
        ]
        if isinstance(cls, _Pending):
            cls.row = row
            self._rows.append(cls)
            heapq.heappush(self._due, (cls.anchor + 2 * self.window_ns, self.event_id, cls))
            self._waiting.setdefault((sym, cls.side, cls.price), []).append(cls)
        else:
            self._rows.append(row)
        self.event_id += 1

    # -- trades table ----------------------------------------------------------

    def trade_rows(self) -> Iterator[list]:
        """Every trade, with what it matched and whether L3 was valid when it was received."""
        ivs: dict[str, list[tuple[int, int]]] = {}
        for iv in self.tracker.intervals:
            ivs.setdefault(iv.symbol, []).append((iv.start_ns, iv.end_ns))
        for v in ivs.values():
            v.sort()
        for t in sorted(self.trades.trades, key=lambda t: (t.ts_ns, t.symbol, t.trade_id)):
            v = ivs.get(t.symbol, [])
            i = bisect.bisect_right(v, (t.recv_ns, float("inf"))) - 1
            l3_valid = i >= 0 and v[i][0] <= t.recv_ns <= v[i][1]
            yield [t.trade_id, t.symbol, t.recv_ns, t.ts_ns, t.taker_side, SIDE[MAKER_SIDE[t.taker_side]],
                   dec(t.price), dec(t.qty), t.ord_type, dec(t.matched), ";".join(map(str, t.event_ids)),
                   int(l3_valid)]
