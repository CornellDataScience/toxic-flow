"""BookQuery: point-in-time order book queries over an L3 event log.

Rules every query follows:

As-of: a query at time t reflects every event with ts <= t, applied in
    original message order (`seq`), and nothing after. Events that share a
    timestamp are applied together, so a query never sees half of a message.

Clock: all times are int ns since the epoch (UTC) in one clock, chosen at load:
    "exchange" (Kraken's per-message `timestamp`, the default) or "recv"
    (local receive time). Strings, datetimes and pd.Timestamps are also
    accepted and converted. Horizons ("100ms", "1s", ...) are in the same clock.

Validity: t is valid only inside a valid interval [start, end] (inclusive),
    and only once the interval's first event (its snapshot) has been applied.
    Outside that, scalar queries return None (book-shaped results) or NaN
    (numbers), and vectorized queries return NaN, or raise InvalidTimeError
    if strict=True. The book is rebuilt from empty at the start of every
    interval and never carried across a gap. mid_after(t, h) is NaN when
    t + h falls outside the interval that contains t.

Depth: Kraken streams only the top `depth` levels per side (100 in our
    recordings). Prices beyond the deepest visible level are unknown, not
    empty; queue_at/queue_ahead warn with OutsideDepthWarning there.

Precision: prices/sizes are exact integers (ticks/lots) internally. Book-shaped
    results are Decimal. mid/spread/best arrays are float64, made only via
    units.ticks_to_float.
"""

import hashlib
import logging
import pickle
import warnings
from bisect import bisect_right
from collections.abc import Hashable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from toxicflow.kraken.book import PairInfo

from .book import BID, EVENT_CODES, Best, Book, PriceLike, side_code
from .units import HorizonLike, TimeLike, TimesLike, decimal_to_ticks, horizon_ns, ticks_to_float, to_ns, to_ns_array

log = logging.getLogger(__name__)

CLOCKS = ("exchange", "recv")
CACHE_VERSION = 1


class InvalidTimeError(LookupError):
    """Query time outside every valid interval (strict mode only)."""


class OutsideDepthWarning(UserWarning):
    """Queried a price beyond the deepest level the feed shows."""


@dataclass(frozen=True)
class BookStep:
    """One step of iter_book.

    book is the library's live book, updated in place between steps: read it,
    or call book.copy() to keep it. events is a positional slice into
    BookQuery.events for the events applied in this step (empty on the first
    step of an interval). reset=True means a new valid interval started and
    any simulator state from before is stale.
    """

    ts: int
    interval: int
    book: Book
    events: slice
    reset: bool


@dataclass
class _Checkpoint:
    ts: int        # as-of time of the stored book
    next_i: int    # first event not yet applied
    book: Book


class BookQuery:
    """Point-in-time queries over one symbol's L3 events. Build with BookQuery.load()."""

    def __init__(self) -> None:
        raise TypeError("use BookQuery.load(...)")

    # ------------------------------------------------------------------ load

    @classmethod
    def load(
        cls,
        events: pd.DataFrame,
        trades: pd.DataFrame | None = None,
        valid_intervals: pd.DataFrame | None = None,
        *,
        pair: PairInfo,
        symbol: str | None = None,
        depth: int | None = None,
        clock: str = "exchange",
        checkpoint_every: int = 2000,
        cache_dir: Path | str | None = None,
        strict: bool = False,
    ) -> "BookQuery":
        """Index an event log for queries.

        events columns:
            ts_<clock> (or ts): int ns; seq: int, original message order
            (strictly increasing); order_id; side: bid/ask; event:
            snapshot/add/modify/delete; price_ticks + qty_lots (ints), or
            price + qty (Decimal/str/float, converted exactly using `pair`).
            qty is the order's remaining size after the event.
            Optional msg_id: events of one exchange message (depth truncation
            happens per message; falls back to per timestamp).
            Each valid interval must begin with the book's snapshot (or an
            empty book): replay starts from nothing at `start`.
        trades: ts_<clock> (or ts), price, qty, side (aggressor). Kept as-is
            for trades_between().
        valid_intervals: start_<clock>/end_<clock> (or start/end), inclusive
            ns. None means one interval spanning all events.
        depth: levels per side the feed shows (enables truncation + depth warnings).
        checkpoint_every: store a full book copy every this many events;
            book_at replays at most this many events.
        cache_dir: if set, the derived top-of-book arrays and checkpoints are
            pickled there, keyed by a hash of the inputs.
        """
        if clock not in CLOCKS:
            raise ValueError(f"clock must be one of {CLOCKS}")
        self = object.__new__(cls)
        self.pair, self.depth, self.clock, self.strict = pair, depth, clock, strict
        self.checkpoint_every = int(checkpoint_every)
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")

        ev = events
        if symbol is not None and "symbol" in ev.columns:
            ev = ev[ev["symbol"] == symbol]
        self.symbol = symbol
        ev = ev.sort_values("seq", kind="stable").reset_index(drop=True)
        self.events = ev
        self._ts = _column(ev, clock, "ts").astype(np.int64)
        if len(self._ts) and np.any(np.diff(self._ts) < 0):
            bad = int(np.argmax(np.diff(self._ts) < 0)) + 1
            raise ValueError(f"event timestamps go backwards in seq order at row {bad}; "
                             "the as-of rule needs them non-decreasing")
        self._kind = ev["event"].map(EVENT_CODES).to_numpy()
        if pd.isna(self._kind).any():
            raise ValueError(f"unknown event types: {set(ev['event'][pd.isna(self._kind)])}")
        self._kind = self._kind.astype(np.int8)
        self._side = np.array([side_code(s) for s in ev["side"]], dtype=np.int8)
        self._price = _ticks(ev, "price", pair.price_precision)
        self._qty = _ticks(ev, "qty", pair.qty_precision)
        self._oid = ev["order_id"].tolist()
        group_key = ev["msg_id"].to_numpy() if "msg_id" in ev.columns else self._ts
        # truncate the book after event i if the next event is another message
        self._msg_end = np.r_[group_key[1:] != group_key[:-1], True] if len(ev) else np.zeros(0, bool)

        if valid_intervals is None:
            starts = self._ts[:1].copy()
            ends = self._ts[-1:].copy()
        else:
            iv = valid_intervals
            if symbol is not None and "symbol" in iv.columns:
                iv = iv[iv["symbol"] == symbol]
            iv = iv.sort_values(_colname(iv, clock, "start"))
            starts = _column(iv, clock, "start").astype(np.int64)
            ends = _column(iv, clock, "end").astype(np.int64)
            if np.any(ends < starts) or np.any(starts[1:] <= ends[:-1]):
                raise ValueError("valid intervals must be non-empty and non-overlapping")
        self._iv_start, self._iv_end = starts, ends
        self._iv_lo = np.searchsorted(self._ts, starts, "left")
        self._iv_hi = np.searchsorted(self._ts, ends, "right")

        self.trades = None
        if trades is not None:
            tr = trades
            if symbol is not None and "symbol" in tr.columns:
                tr = tr[tr["symbol"] == symbol]
            tr_ts = _column(tr, clock, "ts").astype(np.int64)
            order = np.argsort(tr_ts, kind="stable")
            self.trades = tr.iloc[order].reset_index(drop=True)
            self._trade_ts = tr_ts[order]

        cache_path = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"bookquery_{self._cache_key()}.pkl"
            if cache_path.exists():
                try:
                    with open(cache_path, "rb") as fh:
                        self._tob, self._ckpts = pickle.load(fh)
                    log.info("loaded book query cache %s", cache_path)
                    self._ckpt_ts = [[c.ts for c in cs] for cs in self._ckpts]
                    return self
                except (OSError, pickle.UnpicklingError, EOFError, ValueError) as e:
                    log.warning("ignoring unreadable cache %s (%r)", cache_path, e)
        self._build()
        self._ckpt_ts = [[c.ts for c in cs] for cs in self._ckpts]
        if cache_path is not None:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump((self._tob, self._ckpts), fh, protocol=pickle.HIGHEST_PROTOCOL)
                tmp.replace(cache_path)
            except OSError as e:
                log.warning("could not write cache %s (%r)", cache_path, e)
        return self

    def _cache_key(self) -> str:
        h = hashlib.sha256()
        h.update(repr((CACHE_VERSION, self.pair, self.depth, self.clock, self.checkpoint_every)).encode())
        for arr in (self._ts, self._kind, self._side, self._price, self._qty, self._msg_end,
                    self._iv_start, self._iv_end):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(pd.util.hash_pandas_object(self.events["order_id"], index=False).to_numpy().tobytes())
        return h.hexdigest()[:24]

    def _build(self) -> None:
        """One replay of every interval: top-of-book after each timestamp, plus checkpoints."""
        ts, kind, side, price, qty, oid, msg_end = (
            self._ts.tolist(), self._kind.tolist(), self._side.tolist(),
            self._price.tolist(), self._qty.tolist(), self._oid, self._msg_end.tolist())
        rows: list[tuple[int, int, int, int, int, int]] = []
        self._ckpts: list[list[_Checkpoint]] = []
        every = self.checkpoint_every
        for j, (lo, hi) in enumerate(zip(self._iv_lo.tolist(), self._iv_hi.tolist())):
            book = Book(self.pair, self.depth)
            ckpts: list[_Checkpoint] = []
            since = 0
            for i in range(lo, hi):
                book.apply(kind[i], side[i], oid[i], price[i], qty[i])
                if msg_end[i] or i == hi - 1:
                    book.truncate()
                since += 1
                if i == hi - 1 or ts[i + 1] != ts[i]:
                    book.ts = ts[i]
                    rows.append((ts[i], j, *book.best_ticks()))
                    if since >= every:
                        ckpts.append(_Checkpoint(ts[i], i + 1, book.copy()))
                        since = 0
            self._ckpts.append(ckpts)
        arr = np.array(rows, dtype=np.int64).reshape(-1, 6)
        self._tob = {name: arr[:, k].copy() for k, name in
                     enumerate(("ts", "interval", "bid_px", "bid_qty", "ask_px", "ask_qty"))}

    # ------------------------------------------------------------ validity

    def _locate(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """-> (top-of-book row, interval index, valid mask) for each time."""
        j = np.searchsorted(self._iv_start, t, "right") - 1
        ok = j >= 0
        jj = np.where(ok, j, 0)
        if len(self._iv_end):
            ok &= t <= self._iv_end[jj]
        else:
            ok &= False
        idx = np.searchsorted(self._tob["ts"], t, "right") - 1
        ok &= idx >= 0
        ii = np.where(idx >= 0, idx, 0)
        if len(self._tob["ts"]):
            ok &= self._tob["interval"][ii] == jj
        else:
            ok &= False
        return ii, jj, ok

    def _check(self, ok: np.ndarray, t: np.ndarray) -> None:
        if self.strict and not ok.all():
            bad = t[~ok][0]
            raise InvalidTimeError(f"t={int(bad)} ({pd.Timestamp(int(bad))}) is outside every valid interval")

    def is_valid(self, t: TimesLike) -> bool | np.ndarray:
        """Whether the book is known at t (never raises, even in strict mode)."""
        arr = to_ns_array(t)
        ok = self._locate(np.atleast_1d(arr))[2]
        return bool(ok[0]) if arr.ndim == 0 else ok.reshape(arr.shape)

    @property
    def valid_intervals(self) -> list[tuple[int, int]]:
        return list(zip(self._iv_start.tolist(), self._iv_end.tolist()))

    # ------------------------------------------------- top of book (vectorized)

    def _top(self, t: TimesLike) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
        arr = to_ns_array(t)
        flat = np.atleast_1d(arr).ravel()
        ii, _, ok = self._locate(flat)
        self._check(ok, flat)
        return ii, ok, flat, arr.shape

    def _px_series(self, name: str, ii: np.ndarray, ok: np.ndarray) -> np.ndarray:
        raw = self._tob[name][ii]
        good = ok & (raw >= 0)
        prec = self.pair.price_precision if name.endswith("_px") else self.pair.qty_precision
        out = ticks_to_float(raw, prec)
        out[~good] = np.nan
        return out

    def mid_at(self, t: TimesLike) -> float | np.ndarray:
        """(best bid + best ask) / 2 as of t, float. NaN if invalid or a side is empty."""
        ii, ok, _, shape = self._top(t)
        mid = self._mid(ii, ok)
        return float(mid[0]) if shape == () else mid.reshape(shape)

    def _mid(self, ii: np.ndarray, ok: np.ndarray) -> np.ndarray:
        bid, ask = self._tob["bid_px"][ii], self._tob["ask_px"][ii]
        good = ok & (bid >= 0) & (ask >= 0)
        # sum in integer ticks first, so the only rounding is the final float conversion
        mid = ticks_to_float(bid + ask, self.pair.price_precision) / 2
        mid[~good] = np.nan
        return mid

    def spread_at(self, t: TimesLike) -> float | np.ndarray:
        """best ask - best bid as of t, float in price units. NaN if invalid or a side is empty."""
        ii, ok, _, shape = self._top(t)
        bid, ask = self._tob["bid_px"][ii], self._tob["ask_px"][ii]
        good = ok & (bid >= 0) & (ask >= 0)
        out = ticks_to_float(ask - bid, self.pair.price_precision)
        out[~good] = np.nan
        return float(out[0]) if shape == () else out.reshape(shape)

    def best_at(self, t: TimesLike) -> Best | None | pd.DataFrame:
        """Best bid/ask and their total sizes as of t.

        Scalar t -> Best (exact Decimals) or None if invalid.
        Array t -> DataFrame of float columns bid_price, bid_qty, ask_price,
        ask_qty (NaN if invalid or that side is empty).
        """
        ii, ok, flat, shape = self._top(t)
        if shape == ():
            if not ok[0]:
                return None
            i = int(ii[0])
            bp, bq, ap, aq = (int(self._tob[k][i]) for k in ("bid_px", "bid_qty", "ask_px", "ask_qty"))
            book = Book(self.pair)
            return Best(book.px(bp) if bp >= 0 else None, book.qty(bq) if bp >= 0 else None,
                        book.px(ap) if ap >= 0 else None, book.qty(aq) if ap >= 0 else None)
        return pd.DataFrame({
            "bid_price": self._px_series("bid_px", ii, ok),
            "bid_qty": self._px_series("bid_qty", ii, ok & (self._tob["bid_px"][ii] >= 0)),
            "ask_price": self._px_series("ask_px", ii, ok),
            "ask_qty": self._px_series("ask_qty", ii, ok & (self._tob["ask_px"][ii] >= 0)),
        }, index=pd.Index(flat, name="t"))

    def mid_after(
        self, t: TimesLike, horizons: Sequence[HorizonLike] = ("100ms", "1s", "10s")
    ) -> dict[str, float] | pd.DataFrame:
        """Mid at t + h for each horizon, for markouts.

        NaN when t is invalid, or when t + h is past the end of the valid
        interval containing t (never reads a book from across a gap).
        Scalar t -> {horizon: mid}; array t -> DataFrame (one column per
        horizon, plus "mid" at t itself), indexed by t.
        """
        arr = to_ns_array(t)
        flat = np.atleast_1d(arr).ravel()
        ii, jj, ok = self._locate(flat)
        self._check(ok, flat)
        out = {"mid": self._mid(ii, ok)}
        for h in horizons:
            t2 = flat + horizon_ns(h)
            ii2, jj2, ok2 = self._locate(t2)
            out[str(h)] = self._mid(ii2, ok & ok2 & (jj2 == jj))
        if arr.ndim == 0:
            return {k: float(v[0]) for k, v in out.items()}
        return pd.DataFrame(out, index=pd.Index(flat, name="t"))

    # --------------------------------------------------------- full book

    def _book_at(self, t_ns: int, check: bool = True) -> tuple[Book, int, int] | None:
        """-> (fresh book as of t, interval, next event index) or None if invalid."""
        t_arr = np.array([t_ns], dtype=np.int64)
        _, jj, ok = self._locate(t_arr)
        if check:
            self._check(ok, t_arr)
        if not ok[0]:
            return None
        j = int(jj[0])
        k = bisect_right(self._ckpt_ts[j], t_ns) - 1
        if k >= 0:
            ckpt = self._ckpts[j][k]
            book, i = ckpt.book.copy(), ckpt.next_i
        else:
            book, i = Book(self.pair, self.depth), int(self._iv_lo[j])
        i = self._replay(book, i, int(self._iv_hi[j]), t_ns)
        return book, j, i

    def _replay(self, book: Book, i: int, hi: int, t_ns: int) -> int:
        """Apply events i.. with ts <= t_ns (stopping at hi). Returns the next index."""
        stop = min(hi, int(np.searchsorted(self._ts, t_ns, "right")))
        if stop <= i:
            return i
        kind, side, price, qty, msg_end = (a[i:stop].tolist() for a in
                                           (self._kind, self._side, self._price, self._qty, self._msg_end))
        oid = self._oid
        for n in range(stop - i):
            book.apply(kind[n], side[n], oid[i + n], price[n], qty[n])
            if msg_end[n] or i + n == hi - 1:
                book.truncate()
        book.ts = int(self._ts[stop - 1])
        return stop

    def book_at(self, t: TimeLike) -> Book | None:
        """Full L3 book after all events with ts <= t (a private copy), or None if invalid.

        Starts from the nearest checkpoint at or before t and replays forward.
        """
        res = self._book_at(to_ns(t))
        return None if res is None else res[0]

    def depth_at(self, t: TimeLike, levels: int = 10) -> dict[str, list[tuple[Decimal, Decimal]]] | None:
        """{"bids": [(price, size), ...], "asks": [...]}, best first, as of t."""
        book = self.book_at(t)
        return None if book is None else book.depth_levels(levels)

    def _depth_flag(self, book: Book, side: str | int, price: PriceLike) -> None:
        if book.at_depth_edge(side_code(side), book.price_ticks(price)):
            msg = (f"{side} {price} is beyond the {self.depth} levels the feed shows; "
                   "orders there are unknown, not absent")
            if self.strict:
                raise InvalidTimeError(msg)
            warnings.warn(msg, OutsideDepthWarning, stacklevel=3)

    def queue_at(self, side: str, price: PriceLike, t: TimeLike) -> list[tuple[Hashable, Decimal]] | None:
        """Orders at one level as of t: [(order_id, remaining size)], front of the queue first."""
        book = self.book_at(t)
        if book is None:
            return None
        self._depth_flag(book, side, price)
        return book.queue(side, price)

    def queue_ahead(self, side: str, price: PriceLike, t: TimeLike, order_id: Hashable | None = None) -> Decimal | None:
        """Size ahead in the queue at this price as of t (same level only).

        With order_id: size in front of that order (KeyError if it isn't
        resting there). Without: the whole level, i.e. the wait for a new
        order joining now (new orders go to the back).
        """
        book = self.book_at(t)
        if book is None:
            return None
        self._depth_flag(book, side, price)
        return book.queue_ahead(side, price, order_id)

    def iter_book(self, t0: TimeLike, t1: TimeLike) -> Iterator[BookStep]:
        """Walk the book forward through [t0, t1], one step per distinct event timestamp.

        For each valid interval overlapping [t0, t1]: first the book as of
        max(t0, interval start) (reset=True), then one step after each later
        timestamp's events, up to t1. Gaps between intervals are skipped.
        Far cheaper than calling book_at repeatedly.
        """
        a, b = to_ns(t0), to_ns(t1)
        first = max(0, int(np.searchsorted(self._iv_end, a, "left")))
        for j in range(first, len(self._iv_start)):
            start, end = int(self._iv_start[j]), int(self._iv_end[j])
            if start > b:
                break
            lo, hi = int(self._iv_lo[j]), int(self._iv_hi[j])
            if lo >= hi:
                continue
            t_first = max(a, start, int(self._ts[lo]))
            if t_first > min(b, end):
                continue
            res = self._book_at(t_first, check=False)
            if res is None:
                continue
            book, _, i = res
            yield BookStep(int(book.ts), j, book, slice(i, i), True)
            stop = min(hi, int(np.searchsorted(self._ts, min(b, end), "right")))
            while i < stop:
                t_next = int(self._ts[i])
                n = self._replay(book, i, hi, t_next)
                yield BookStep(t_next, j, book, slice(i, n), False)
                i = n

    # --------------------------------------------------------------- trades

    def trades_between(self, t0: TimeLike, t1: TimeLike) -> pd.DataFrame:
        """Trades with t0 < ts <= t1 (same clock), in time order."""
        if self.trades is None:
            raise ValueError("no trades were loaded")
        lo = np.searchsorted(self._trade_ts, to_ns(t0), "right")
        hi = np.searchsorted(self._trade_ts, to_ns(t1), "right")
        return self.trades.iloc[lo:hi]

    # -------------------------------------------------------------- helpers

    def price_to_float(self, ticks: int | np.ndarray) -> float | np.ndarray:
        return ticks_to_float(ticks, self.pair.price_precision)

    def qty_to_float(self, lots: int | np.ndarray) -> float | np.ndarray:
        return ticks_to_float(lots, self.pair.qty_precision)


def _colname(df: pd.DataFrame, clock: str, base: str) -> str:
    for name in (f"{base}_{clock}", base):
        if name in df.columns:
            return name
    raise KeyError(f"need a {base}_{clock!s} or {base} column; have {list(df.columns)}")


def _column(df: pd.DataFrame, clock: str, base: str) -> np.ndarray:
    return df[_colname(df, clock, base)].to_numpy()


def _ticks(df: pd.DataFrame, base: str, precision: int) -> np.ndarray:
    exact = {"price": "price_ticks", "qty": "qty_lots"}[base]
    if exact in df.columns:
        return df[exact].to_numpy(dtype=np.int64)
    return np.array([decimal_to_ticks(v, precision) for v in df[base]], dtype=np.int64)
