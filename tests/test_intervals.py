from decimal import Decimal

from toxicflow.kraken.book import L3Book, PairInfo
from toxicflow.kraken.intervals import find_intervals

SYM = "BTC/USD"
META = {"stream": "l3", "depth": 10, "pairs": {SYM: {"price_precision": 1, "qty_precision": 8}}}
S = 1_000_000_000  # ns per second


class Feed:
    """Builds recorded L3 records with correct checksums (or deliberately wrong ones)."""

    def __init__(self):
        self.shadow = L3Book(PairInfo(SYM, 1, 8), 10)
        self.records = []
        self.n = 0

    def meta(self, t):
        self.records.append((t, "meta", META))

    def status(self, t, conn=1):
        self.records.append((t, "msg", {"channel": "status", "type": "update", "data": [{"connection_id": conn}]}))

    def heartbeat(self, t):
        self.records.append((t, "msg", {"channel": "heartbeat"}))

    def _book_msg(self, t, msg_type, bids, bad):
        item = {"symbol": SYM, "bids": bids, "asks": [], "checksum": 0}
        self.shadow.apply(msg_type, dict(item))
        self.shadow.synced = True
        item["checksum"] = self.shadow.checksum() ^ (1 if bad else 0)
        self.records.append((t, "msg", {"channel": "level3", "type": msg_type, "data": [item]}))

    def snapshot(self, t, bad=False):
        self.shadow.reset()
        self._book_msg(t, "snapshot", [{"order_id": "B0", "limit_price": Decimal("100.0"), "order_qty": Decimal("1")}], bad)

    def add(self, t, bad=False):
        self.n += 1
        ev = {"event": "add", "order_id": f"B{self.n}", "limit_price": Decimal("99.0") - self.n % 5, "order_qty": Decimal("1")}
        self._book_msg(t, "update", [ev], bad)


def run(feed, gap_s=5.0):
    return find_intervals(sorted(feed.records, key=lambda r: r[0]), gap_s)


def test_clean_stream_is_one_interval():
    f = Feed()
    f.meta(0)
    f.status(1)
    f.snapshot(2 * S)
    for i in range(3, 20):
        f.add(i * S)
    t = run(f)
    assert [(iv.start_ns, iv.end_ns, iv.start_reason, iv.end_reason) for iv in t.intervals] == [
        (2 * S, 19 * S, "snapshot", "end_of_data")
    ]
    st = t.stats[SYM]
    assert st.checksum_fail == 0 and st.desync_periods == 0 and st.checksum_pass_rate == 1.0


def test_checksum_failure_ends_interval_until_next_snapshot():
    f = Feed()
    f.meta(0)
    f.status(0)
    f.snapshot(1 * S)
    f.add(2 * S)
    f.add(3 * S, bad=True)  # desync
    f.add(4 * S)            # ignored: book unsynced
    f.snapshot(5 * S)       # resync
    f.add(6 * S)
    t = run(f)
    assert [(iv.start_ns, iv.end_ns, iv.end_reason) for iv in t.intervals] == [
        (1 * S, 2 * S, "checksum_fail"),
        (5 * S, 6 * S, "end_of_data"),
    ]
    st = t.stats[SYM]
    assert st.desync_periods == 1 and st.desync_s == [2.0] and st.skipped_unsynced == 1


def test_gap_breaks_interval_and_resumes():
    f = Feed()
    f.meta(0)
    f.status(0)
    f.snapshot(1 * S)
    f.add(2 * S)
    f.add(20 * S)  # 18 s of silence
    f.add(21 * S)
    t = run(f)
    assert [(iv.start_ns, iv.end_ns, iv.start_reason, iv.end_reason) for iv in t.intervals] == [
        (1 * S, 2 * S, "snapshot", "gap"),
        (20 * S, 21 * S, "resume_after_gap", "end_of_data"),
    ]
    assert t.longest_gap_s == 18.0 and len(t.gaps) == 1


def test_heartbeats_keep_connection_alive_without_book_updates():
    f = Feed()
    f.meta(0)
    f.status(0)
    f.snapshot(1 * S)
    for i in range(2, 30):
        f.heartbeat(i * S)
    f.add(30 * S)
    t = run(f)
    assert len(t.intervals) == 1 and t.gaps == []
    assert t.stats[SYM].longest_update_gap_s == 29.0


def test_reconnect_and_file_boundary_need_fresh_snapshot():
    f = Feed()
    f.meta(0)
    f.status(0)
    f.snapshot(1 * S)
    f.add(2 * S)
    f.status(3 * S, conn=2)  # reconnect
    f.snapshot(4 * S)
    f.add(5 * S)
    f.meta(6 * S)    # hourly rotation
    f.add(7 * S)     # ignored: new session has no snapshot yet
    f.snapshot(8 * S)
    t = run(f)
    assert [(iv.start_ns, iv.end_ns, iv.end_reason) for iv in t.intervals] == [
        (1 * S, 2 * S, "reconnect"),  # ends at the last message before the new connection
        (4 * S, 5 * S, "file_boundary"),
        (8 * S, 8 * S, "end_of_data"),
    ]
    assert t.reconnects == 1 and t.sessions == 2


def test_reconnect_after_file_rotation_is_detected():
    # Rotation keeps the same connection (no new status); a later status with a
    # new connection_id in the next file is still a reconnect, not a resnapshot.
    f = Feed()
    f.meta(0)
    f.status(0, conn=1)
    f.snapshot(1 * S)
    f.meta(2 * S)  # rotation
    f.snapshot(3 * S)
    f.add(4 * S)
    f.status(6 * S, conn=2)
    f.snapshot(7 * S)
    t = run(f)
    assert [iv.end_reason for iv in t.intervals] == ["file_boundary", "reconnect", "end_of_data"]
    assert t.reconnects == 1
