"""Find the time intervals where a recorded L3 book is trustworthy.

Replays recorded L3 messages through L3Book and tracks, per symbol, the
stretches of time where the local book is known to match Kraken's. Time
counts as valid only after a snapshot whose checksum matches, and only while
every later checksum also matches.

An interval ends (reason in parentheses) when:
  - a checksum fails (checksum_fail): the interval ends at that symbol's last
    good message. The symbol stays invalid until a fresh snapshot, and that
    whole stretch counts as one "desync period".
  - the connection goes silent for longer than gap_s (gap). Heartbeats arrive
    about once a second, so any silence is missing data. The interval ends at
    the last message before the silence. If the book is still in sync, a new
    interval starts at the next matching checksum (start reason resume_after_gap).
  - the websocket reconnects (reconnect): a `status` message arrives with a
    new connection_id, and the recorder resets its books and resubscribes.
  - a new file / writer session starts (file_boundary): every meta line means
    the recorder resubscribed, so a fresh snapshot follows.
  - a new snapshot arrives while the book was valid (resnapshot).
  - the data ends (end_of_data).

All times are local receive times (recv_ns).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field

from .book import L3Book, PairInfo


@dataclass
class Interval:
    symbol: str
    start_ns: int
    end_ns: int
    start_reason: str
    end_reason: str
    updates: int = 0  # book items with a passing checksum inside the interval

    @property
    def duration_s(self) -> float:
        return (self.end_ns - self.start_ns) / 1e9


@dataclass
class SymbolStats:
    snapshots: int = 0
    updates: int = 0
    checksum_ok: int = 0
    checksum_fail: int = 0
    skipped_unsynced: int = 0  # updates that arrived while the book had no valid snapshot
    desync_periods: int = 0
    desync_s: list[float] = field(default_factory=list)  # each desync: failure -> next valid start
    longest_update_gap_s: float = 0.0  # longest silence for this symbol *inside* a valid interval
    longest_update_gap_at_ns: int | None = None

    @property
    def checksum_pass_rate(self) -> float | None:
        checked = self.checksum_ok + self.checksum_fail
        return self.checksum_ok / checked if checked else None


@dataclass
class _Open:
    start_ns: int
    start_reason: str
    last_ok_ns: int
    updates: int = 0


class IntervalTracker:
    """Feed it (recv_ns, kind, payload) records in time order, then call finish()."""

    def __init__(self, gap_s: float = 5.0):
        self.gap_ns = int(gap_s * 1e9)
        self.intervals: list[Interval] = []
        self.stats: dict[str, SymbolStats] = {}
        self.gaps: list[tuple[int, float]] = []  # (start recv_ns, seconds) for gaps > gap_s
        self.longest_gap_s = 0.0
        self.longest_gap_at_ns: int | None = None
        self.reconnects = 0
        self.sessions = 0  # meta lines seen (one per file / writer restart)
        self.first_ns: int | None = None
        self.last_ns: int | None = None
        self._meta: dict | None = None
        self._books: dict[str, L3Book] = {}
        self._open: dict[str, _Open] = {}
        self._desync_since: dict[str, int] = {}
        self._last_update_ns: dict[str, int] = {}
        self._connection_id = None
        self._after_gap: set[str] = set()

    # -- public API ---------------------------------------------------------

    def feed(self, recv_ns: int, kind: str, payload: dict) -> None:
        if self.first_ns is None:
            self.first_ns = recv_ns
        if self.last_ns is not None:
            gap_s = (recv_ns - self.last_ns) / 1e9
            if gap_s > self.longest_gap_s:
                self.longest_gap_s, self.longest_gap_at_ns = gap_s, self.last_ns
            if recv_ns - self.last_ns > self.gap_ns:
                self.gaps.append((self.last_ns, gap_s))
                self._close_all(self.last_ns, "gap")
                self._after_gap = set(self._books)

        if kind == "meta":
            self._on_meta(payload)
        else:
            self._on_msg(recv_ns, payload)
        self.last_ns = recv_ns

    def finish(self) -> None:
        if self.last_ns is not None:
            self._close_all(self.last_ns, "end_of_data")

    @property
    def span_s(self) -> float:
        if self.first_ns is None:
            return 0.0
        return (self.last_ns - self.first_ns) / 1e9

    # -- record handling ----------------------------------------------------

    def _on_meta(self, meta: dict) -> None:
        if meta.get("stream") not in (None, "l3"):
            raise ValueError(f"expected an l3 recording, got stream={meta.get('stream')!r}")
        if self.last_ns is not None:
            self._close_all(self.last_ns, "file_boundary")
        self._meta = meta
        self._books.clear()
        self.sessions += 1

    def _on_msg(self, recv_ns: int, msg: dict) -> None:
        channel = msg.get("channel")
        if channel == "status":
            # Kraken sends `status` once per connection. A new connection_id means
            # we reconnected: the recorder reset its books and is resubscribing.
            # (Files rotate on the same connection, so don't reset this on meta.)
            conn = (msg.get("data") or [{}])[0].get("connection_id")
            if self._connection_id is not None and conn != self._connection_id:
                self.reconnects += 1
                self._close_all(self.last_ns, "reconnect")
                for book in self._books.values():
                    book.reset()
            self._connection_id = conn
            return
        if channel != "level3" or msg.get("type") not in ("snapshot", "update"):
            return
        if self._meta is None:
            raise ValueError("level3 message before any meta line; can't get pair precision")
        for item in msg["data"]:
            self._on_book_item(recv_ns, msg["type"], item)

    def _on_book_item(self, recv_ns: int, msg_type: str, item: dict) -> None:
        sym = item["symbol"]
        st = self.stats.setdefault(sym, SymbolStats())
        book = self._books.get(sym)
        if book is None:
            p = self._meta["pairs"][sym]
            book = self._books[sym] = L3Book(PairInfo(sym, p["price_precision"], p["qty_precision"]), self._meta["depth"])

        if msg_type == "snapshot":
            st.snapshots += 1
            if sym in self._open:
                self._close(sym, self._open[sym].last_ok_ns, "resnapshot")
        else:
            st.updates += 1

        ok = book.apply(msg_type, item)
        if ok is None:
            st.skipped_unsynced += 1
            return
        if ok is False:
            st.checksum_fail += 1
            st.desync_periods += 1
            if sym in self._open:
                self._close(sym, self._open[sym].last_ok_ns, "checksum_fail")
            self._desync_since[sym] = recv_ns
            return

        st.checksum_ok += 1
        cur = self._open.get(sym)
        if cur is None:
            if msg_type == "snapshot":
                reason = "snapshot"
            elif sym in self._after_gap:
                reason = "resume_after_gap"
            else:  # can't happen: a synced book with no open interval had a gap or a failure
                reason = "resume"
            self._open[sym] = cur = _Open(recv_ns, reason, recv_ns)
            self._after_gap.discard(sym)
            since = self._desync_since.pop(sym, None)
            if since is not None:
                st.desync_s.append((recv_ns - since) / 1e9)
        else:
            prev = self._last_update_ns.get(sym, cur.start_ns)
            g = (recv_ns - prev) / 1e9
            if g > st.longest_update_gap_s:
                st.longest_update_gap_s, st.longest_update_gap_at_ns = g, prev
        cur.last_ok_ns = recv_ns
        cur.updates += 1
        self._last_update_ns[sym] = recv_ns

    # -- interval bookkeeping -----------------------------------------------

    def _close(self, sym: str, end_ns: int, reason: str) -> None:
        cur = self._open.pop(sym)
        self._last_update_ns.pop(sym, None)
        self.intervals.append(Interval(sym, cur.start_ns, max(end_ns, cur.start_ns), cur.start_reason, reason, cur.updates))

    def _close_all(self, end_ns: int, reason: str) -> None:
        for sym in list(self._open):
            self._close(sym, end_ns, reason)


def find_intervals(records: Iterable[tuple[int, str, dict]], gap_s: float = 5.0) -> IntervalTracker:
    tracker = IntervalTracker(gap_s)
    for rec in records:
        tracker.feed(*rec)
    tracker.finish()
    return tracker
