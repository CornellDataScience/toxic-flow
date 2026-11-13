"""Trade prints as execution evidence for L3 reductions (design §6.5.2, §6.5.4, §6.5.5, §8.7; Phase 3).

What the recordings support (measured on 2026-10-18 hours, see the Stage 5 notes):

* The public stream (data/raw/public) records Kraken's v2 `trade` channel: symbol, `side`
  (the TAKER side), price, qty, ord_type, trade_id (contiguous per book) and an exchange
  timestamp with microsecond digits. There is no maker order ID, so a trade is never tied
  to a resting order directly; attribution is by matching.
* A trade's timestamp equals the L3 *item* timestamp of the reduction it caused, truncated to
  the microsecond (1524 of 1525 trades on hour 14; 0 on the wrong book side). The per-event
  `timestamp` is not usable: for deletes it is often the order's original time. Multi-level
  sweeps sometimes stamp later prints 1 µs after the item, hence the default `after_ns`.
* Within one (symbol, book side, price, µs) key, trades and reductions line up one-to-one in
  FIFO order (1118 of 1130 keys on hour 14).

Attribution consumes trade volume once, in L3 stream order, across EVERY reduction that could
have been an execution (a quantity reduction at the touch, the same test QueueTracker uses),
not only the ones that reach a simulated order; otherwise a cancel could borrow volume that
belongs to a real fill. Each reduction is labeled once (never per order) as slices:

    TRADE_CONFIRMED  covered by trades on the maker side at the same price stamped in the item's µs
    TRADE_PLAUSIBLE  covered by such trades stamped within the tolerance window but not the same µs
    TRADE_ABSENT     not covered, while the trade feed is healthy: treated as a cancel
    L3 label         not covered and the feed is unhealthy there (no coverage, trade_id gap,
                     unusable or conflicting print): L3_MODIFY_DOWN / L3_DELETE_AMBIGUOUS

Trades are loaded once (TradeIndex) and never reach a strategy, so no future information can
enter a decision (§6.5.5).
"""

import bisect
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from toxicflow.kraken.book import PairInfo

from .config import SimConfig
from .execution import l3_evidence
from .models import Evidence, EvidencePart, Mechanism, Side
from .reader import KIND_GAP, KIND_META, KIND_MSG, KIND_TRUNCATED, RecordingReader, StreamRecord, book_items
from .sim_book import BookEvent, SimL3Book
from .timeutil import parse_rfc3339_ns

US = 1_000


@dataclass(frozen=True, slots=True)
class MatchTolerance:
    """Where a trade may be stamped relative to the reduction's item time truncated to the µs.

    Defaults are measured, not guessed: same µs for 99% of trades, +1 µs for later prints of a
    multi-level sweep, none earlier. `before_ns`/`after_ns` widen the window; only an exact µs
    match is ever TRADE_CONFIRMED.
    """

    before_ns: int = 0
    after_ns: int = 1 * US

    def __post_init__(self):
        for name in ("before_ns", "after_ns"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int) or v < 0:
                raise ValueError(f"{name} must be an int >= 0, got {v!r}")


DEFAULT_TOLERANCE = MatchTolerance()


@dataclass(frozen=True, slots=True)
class Trade:
    """One trade print, observed fields only."""

    symbol: str
    taker_side: Side | None  # Kraken `side` is the taker's; None when missing or not buy/sell
    price: Decimal
    qty: Decimal
    trade_id: int | None
    ts_ns: int  # exchange timestamp (µs precision on the wire)
    ord_type: str | None
    ref: tuple[int, int]  # (StreamRecord.seq, index in the message) in the public stream

    @property
    def maker_book_side(self) -> str | None:
        """Book side of the resting order it executed against (derived from the taker side)."""
        if self.taker_side is None:
            return None
        return "bids" if self.taker_side is Side.SELL else "asks"


@dataclass
class TradeIndex:
    """All trades for the simulated symbols, sorted per symbol, plus where the feed is healthy.

    A symbol is healthy at time t when t lies inside a stretch of the public stream with no
    session restart, receive gap or truncation, and no trade_id gap, unusable print or
    conflicting duplicate falls near t.
    """

    trades: dict[str, list[Trade]]
    segments: list[tuple[int, int]]  # exchange-time spans of the public stream between breaks
    unhealthy: dict[str, list[tuple[int, int]]]  # per symbol, closed intervals
    diag: Counter
    tolerance: MatchTolerance = DEFAULT_TOLERANCE

    @classmethod
    def from_files(cls, files: Sequence[Path], symbols: Iterable[str], gap_ns: int | None = None,
                   tolerance: MatchTolerance = DEFAULT_TOLERANCE) -> "TradeIndex":
        kw = {} if gap_ns is None else {"gap_ns": gap_ns}
        return cls.from_records(RecordingReader(files=files, stream="public", **kw), symbols, tolerance)

    @classmethod
    def from_records(cls, records: Iterable[StreamRecord], symbols: Iterable[str],
                     tolerance: MatchTolerance = DEFAULT_TOLERANCE) -> "TradeIndex":
        symbols = frozenset(symbols)
        diag: Counter = Counter()
        by_id: dict[tuple[str, int], Trade] = {}
        trades: dict[str, list[Trade]] = defaultdict(list)
        bad: dict[str, list[int]] = defaultdict(list)  # times of prints that make the feed unhealthy
        segments: list[tuple[int, int]] = []
        seg: list[int] = []

        def close():
            if seg:
                segments.append((min(seg), max(seg)))
                seg.clear()

        for rec in records:
            if rec.kind in (KIND_META, KIND_GAP, KIND_TRUNCATED):
                close()
                continue
            if rec.kind != KIND_MSG:
                continue
            msg = rec.payload
            channel = msg.get("channel")
            if channel not in ("trade", "book") or not isinstance(msg.get("data"), list):
                continue
            for i, item in enumerate(msg["data"]):
                try:
                    ts = parse_rfc3339_ns(item.get("timestamp"))
                except ValueError:
                    diag["bad_timestamp"] += 1
                    continue
                seg.append(ts)
                if channel != "trade" or item.get("symbol") not in symbols:
                    continue
                t = _parse_trade(item, ts, (rec.seq, i), diag)
                if t is None:
                    bad[item["symbol"]].append(ts)
                    continue
                if t.taker_side is None:
                    bad[t.symbol].append(ts)  # it executed against one side or the other: unknowable
                if t.trade_id is not None:
                    key = (t.symbol, t.trade_id)
                    seen = by_id.get(key)
                    if seen is not None:
                        same = (seen.taker_side, seen.price, seen.qty, seen.ts_ns) == (
                            t.taker_side, t.price, t.qty, t.ts_ns)
                        diag["duplicate_trade_id" if same else "conflicting_trade_id"] += 1
                        if not same:
                            bad[t.symbol] += [seen.ts_ns, t.ts_ns]
                        continue
                    by_id[key] = t
                else:
                    diag["missing_trade_id"] += 1
                trades[t.symbol].append(t)
                diag["trades"] += 1
        close()

        unhealthy: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for sym, ts_list in trades.items():
            ts_list.sort(key=lambda t: (t.ts_ns, t.trade_id is None, t.trade_id or 0, t.ref))
            by_trade_id = sorted(ts_list, key=lambda t: (t.trade_id is None, t.trade_id or 0))
            ids = [t for t in by_trade_id if t.trade_id is not None]
            for a, b in zip(ids, ids[1:]):
                if b.trade_id != a.trade_id + 1:
                    diag["trade_id_gaps"] += 1
                    unhealthy[sym].append((min(a.ts_ns, b.ts_ns), max(a.ts_ns, b.ts_ns)))
                if b.ts_ns < a.ts_ns:
                    diag["ts_regressions"] += 1
            for t in ts_list:
                if t.trade_id is None:  # continuity cannot be verified around it
                    unhealthy[sym].append((t.ts_ns, t.ts_ns))
        for sym, times in bad.items():
            unhealthy[sym] += [(ts, ts) for ts in times]
        for sym in unhealthy:
            unhealthy[sym] = _merge(unhealthy[sym])
        return cls(dict(trades), sorted(segments), dict(unhealthy), diag, tolerance)

    def covered(self, ts_ns: int) -> bool:
        i = bisect.bisect_right(self.segments, (ts_ns, float("inf"))) - 1
        return i >= 0 and self.segments[i][0] <= ts_ns <= self.segments[i][1]

    def healthy(self, symbol: str, item_ts_ns: int) -> bool:
        """Can the absence of a matching trade at this item time be trusted?"""
        us = item_ts_ns // US * US
        lo, hi = us - self.tolerance.before_ns, us + self.tolerance.after_ns
        if not (self.covered(lo) and self.covered(hi)):
            return False
        # a bad print at time τ could have matched any item whose window [lo, hi] contains τ
        for a, b in self.unhealthy.get(symbol, ()):
            if a <= hi and b >= lo:
                return False
        return True


def _parse_trade(item: dict, ts: int, ref: tuple[int, int], diag: Counter) -> Trade | None:
    try:
        price, qty = Decimal(str(item["price"])), Decimal(str(item["qty"]))
    except (KeyError, ArithmeticError, ValueError):
        diag["unusable_trade"] += 1
        return None
    if not (price.is_finite() and qty.is_finite() and price > 0 and qty > 0):
        diag["unusable_trade"] += 1
        return None
    side = {"buy": Side.BUY, "sell": Side.SELL}.get(item.get("side"))
    if side is None:
        diag["invalid_side"] += 1
    tid = item.get("trade_id")
    tid = tid if isinstance(tid, int) and not isinstance(tid, bool) else None
    return Trade(item["symbol"], side, price, qty, tid, ts, item.get("ord_type"), ref)


def _merge(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


# --- attribution ------------------------------------------------------------------------------------------


def is_plausible_execution(ev: BookEvent) -> bool:
    """A quantity reduction at the touch: the only kind of event that can be an execution
    (the same structural test as QueueTracker._reach, without the per-order part)."""
    if ev.dropped_by_truncation or ev.kind not in ("modify", "delete"):
        return False
    if ev.kind == "modify" and (ev.requeued or ev.price_after != ev.price_before):
        return False
    if ev.reduction <= 0:
        return False
    best = ev.best_bid_before if ev.side == "bids" else ev.best_ask_before
    return ev.price_before == best


@dataclass(frozen=True, slots=True)
class AttributedReduction:
    order_id: str  # the historical (maker) order, observed in L3
    kind: str
    side: str
    price: Decimal
    volume: Decimal
    parts: tuple[EvidencePart, ...]


@dataclass
class TradeAttribution:
    """Labels for every plausible execution in the trusted L3 stream, keyed by (seq, event_index)."""

    by_ref: dict[tuple[int, int], AttributedReduction]
    trade_used: dict[tuple[str, tuple[int, int]], Decimal]  # (symbol, trade ref) -> volume attributed
    stats: dict = field(default_factory=dict)
    tolerance: MatchTolerance = DEFAULT_TOLERANCE

    def parts_for(self, ref: tuple, ev: BookEvent, volume: Decimal, mechanism: Mechanism) -> tuple[EvidencePart, ...]:
        """Evidence slices for a reach. CROSS_ARRIVAL never has a historical print (the arriving order
        rested), so it keeps its L3 label."""
        if mechanism is Mechanism.CROSS_ARRIVAL:
            return (EvidencePart(volume, Evidence.L3_CROSS_ARRIVAL),)
        rec = self.by_ref.get(ref)
        if rec is None or rec.order_id != ev.order_id or rec.volume != volume:
            raise ValueError(f"no attribution for reach {ref} ({ev.kind} {ev.order_id}): the attribution "
                             "pass must read the same L3 files with the engine's symbols and gap_ns")
        return rec.parts


class _Pools:
    """Remaining (unattributed) volume of every trade, looked up by maker side, price and time."""

    def __init__(self, index: TradeIndex):
        self.index = index
        self.keys: dict[tuple[str, str, Decimal], list[int]] = {}
        self.items: dict[tuple[str, str, Decimal], list[Trade]] = defaultdict(list)
        self.left: dict[tuple[str, tuple[int, int]], Decimal] = {}
        for sym, ts in index.trades.items():
            for t in ts:
                if t.maker_book_side is None:
                    continue
                self.items[(sym, t.maker_book_side, t.price)].append(t)
                self.left[(sym, t.ref)] = t.qty
        for k, v in self.items.items():
            self.keys[k] = [t.ts_ns for t in v]

    def window(self, symbol: str, side: str, price: Decimal, lo: int, hi: int) -> list[Trade]:
        key = (symbol, side, price)
        times = self.keys.get(key)
        if not times:
            return []
        lst = self.items[key]
        return [t for t in lst[bisect.bisect_left(times, lo):bisect.bisect_right(times, hi)]
                if self.left[(symbol, t.ref)] > 0]


def attribute_trades(cfg: SimConfig, index: TradeIndex, files: Sequence[Path] | None = None,
                     records: Iterable[StreamRecord] | None = None) -> TradeAttribution:
    """One pass over the L3 recording: label every plausible execution in checksum-valid items."""
    if (files is None) == (records is None):
        raise ValueError("give exactly one of files or records")
    if records is None:  # exactly what HistoricalReplayEngine.run reads, so seq (and refs) agree
        records = RecordingReader(files=files, symbols=cfg.symbols, gap_ns=cfg.gap_ns)
    tol = index.tolerance
    pools = _Pools(index)
    books: dict[str, SimL3Book] = {}
    meta: dict | None = None
    by_ref: dict[tuple[int, int], AttributedReduction] = {}
    red_count: Counter = Counter()
    red_qty: dict = defaultdict(Decimal)
    l3_span: dict[str, list[int]] = {}
    for rec in records:
        if rec.kind == KIND_META:
            meta = rec.payload
            books.clear()
            continue
        if rec.kind == KIND_TRUNCATED:
            books.clear()
            continue
        for msg_type, item in book_items(rec):
            sym = item.get("symbol")
            if sym not in cfg.symbols:
                continue
            book = books.get(sym)
            if book is None:
                if meta is None or sym not in meta.get("pairs", {}):
                    continue
                p = meta["pairs"][sym]
                book = books[sym] = SimL3Book(PairInfo(sym, p["price_precision"], p["qty_precision"]), meta["depth"])
            ok = book.apply(msg_type, item, position=rec)
            if msg_type != "update" or ok is not True:
                continue  # only events of a checksum-valid item are trusted (R8, I7)
            try:
                item_ts = parse_rfc3339_ns(item.get("timestamp"))
            except ValueError:
                item_ts = None
            if item_ts is not None:
                span = l3_span.setdefault(sym, [item_ts, item_ts])
                span[0], span[1] = min(span[0], item_ts), max(span[1], item_ts)
            for ev in book.last_events:
                if not is_plausible_execution(ev):
                    continue
                parts = _attribute(ev, sym, item_ts, index, pools, tol)
                by_ref[(rec.seq, ev.event_index)] = AttributedReduction(
                    ev.order_id, ev.kind, ev.side, ev.price_before, ev.reduction, parts)
                for part in parts:
                    red_count[(sym, ev.kind, part.evidence.name)] += 1
                    red_qty[(sym, ev.kind, part.evidence.name)] += part.qty
    used = {(sym, t.ref): t.qty - pools.left[(sym, t.ref)]
            for sym, ts in index.trades.items() for t in ts if (sym, t.ref) in pools.left}
    stats = _stats(index, pools, used, red_count, red_qty, l3_span)
    return TradeAttribution(by_ref, used, stats, tol)


def _attribute(ev: BookEvent, sym: str, item_ts: int | None, index: TradeIndex, pools: _Pools,
               tol: MatchTolerance) -> tuple[EvidencePart, ...]:
    volume = ev.reduction
    fallback = l3_evidence(Mechanism.LEVEL, ev.kind)  # MODIFY_DOWN or DELETE_AMBIGUOUS
    if item_ts is None:
        return (EvidencePart(volume, fallback),)
    us = item_ts // US * US
    exact_qty, exact_ids, near_qty, near_ids = Decimal(0), [], Decimal(0), []
    remaining = volume
    for t in pools.window(sym, ev.side, ev.price_before, us - tol.before_ns, us + tol.after_ns):
        key = (sym, t.ref)
        take = min(remaining, pools.left[key])
        pools.left[key] -= take
        remaining -= take
        if t.ts_ns == us:
            exact_qty += take
            exact_ids.append(t.trade_id)
        else:
            near_qty += take
            near_ids.append(t.trade_id)
        if remaining == 0:
            break
    parts = []
    if exact_qty:
        parts.append(EvidencePart(exact_qty, Evidence.TRADE_CONFIRMED, tuple(i for i in exact_ids if i is not None)))
    if near_qty:
        parts.append(EvidencePart(near_qty, Evidence.TRADE_PLAUSIBLE, tuple(i for i in near_ids if i is not None)))
    if remaining:
        parts.append(EvidencePart(remaining, Evidence.TRADE_ABSENT if index.healthy(sym, item_ts) else fallback))
    return tuple(parts)


def _stats(index, pools, used, red_count, red_qty, l3_span) -> dict:
    """Attribution accounting: every trade and every plausible execution lands in exactly one row."""
    trades: Counter = Counter()
    vol: dict = defaultdict(Decimal)
    for sym, ts in index.trades.items():
        span = l3_span.get(sym)
        for t in ts:
            if t.maker_book_side is None:
                cls = "INVALID_SIDE"
            elif span is None or not span[0] <= t.ts_ns <= span[1]:
                cls = "OUTSIDE_L3"
            else:
                u = used.get((sym, t.ref), Decimal(0))
                cls = "MATCHED" if u == t.qty else "UNMATCHED" if u == 0 else "PARTIAL"
            trades[(sym, cls)] += 1
            vol[(sym, cls)] += t.qty
            vol[(sym, "UNATTRIBUTED_QTY")] += t.qty - used.get((sym, t.ref), Decimal(0))
    return {
        "trade_feed": dict(index.diag),
        "trades": {f"{s}|{c}": n for (s, c), n in sorted(trades.items())},
        "trade_qty": {f"{s}|{c}": q for (s, c), q in sorted(vol.items())},
        "reductions": {f"{s}|{k}|{e}": n for (s, k, e), n in sorted(red_count.items())},
        "reduction_qty": {f"{s}|{k}|{e}": q for (s, k, e), q in sorted(red_qty.items())},
    }
