"""Position-stamped, in-order iteration over hourly recordings.

Yields StreamRecords for one stream across files sorted by name (= time). `seq`
is the clock: records are never sorted by timestamp. Besides the recorded
`meta` and `msg` lines it emits two synthetic markers so the replay engine sees
every discontinuity explicitly:

    gap        recv_ns jumped by more than `gap_ns` since the previous message
    truncated  the file ended early or its tail was corrupt; the rest of that
               file is skipped and iteration continues with the next file

read_recording() swallows EOFError/JSONDecodeError silently, which would hide a
truncated tail, so the line loop is repeated here with the same parsing.
"""

import gzip
import json
import zlib
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from toxicflow.kraken.config import DATA_DIR
from toxicflow.kraken.ws import loads

KIND_META = "meta"
KIND_MSG = "msg"
KIND_GAP = "gap"
KIND_TRUNCATED = "truncated"

DEFAULT_GAP_NS = 5_000_000_000  # same default as scripts/validate_recording.py

_SUFFIX = ".jsonl.gz"
# A killed writer leaves a short gzip member, a partial line, or (when restarted
# in the same hour) a new member glued onto garbage.
_CORRUPT_TAIL = (EOFError, gzip.BadGzipFile, zlib.error, UnicodeDecodeError, json.JSONDecodeError, KeyError)


@dataclass(frozen=True, slots=True)
class StreamRecord:
    file: Path
    line_no: int  # 1-based line in `file` (for markers: the line they precede)
    seq: int  # global position over everything this reader yields
    recv_ns: int  # local receive clock; only meaningful for gap detection
    kind: str  # KIND_*
    payload: dict

    @property
    def is_meta(self) -> bool:
        return self.kind == KIND_META


def file_hour(path: Path, stream: str) -> datetime:
    stamp = path.name.removeprefix(f"{stream}_").removesuffix(_SUFFIX)
    return datetime.strptime(stamp, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)


def list_files(
    root: Path, stream: str, start: datetime | None = None, end: datetime | None = None
) -> list[Path]:
    """Hourly files of `stream` whose hour overlaps [start, end), oldest first."""
    hours = []
    for path in Path(root, stream).glob(f"*/{stream}_*{_SUFFIX}"):
        hour = file_hour(path, stream)
        if (start is None or hour + timedelta(hours=1) > start) and (end is None or hour < end):
            hours.append((hour, path))
    return [p for _, p in sorted(hours)]


class RecordingReader:
    def __init__(
        self,
        root: Path | None = None,
        stream: str = "l3",
        symbols: Iterable[str] | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        gap_ns: int = DEFAULT_GAP_NS,
        files: Iterable[Path] | None = None,
    ):
        """`start`/`end` are tz-aware datetimes selecting files by hour. `files` overrides discovery."""
        self.stream = stream
        self.symbols = frozenset(symbols) if symbols else None
        self.gap_ns = gap_ns
        root = Path(root) if root is not None else DATA_DIR / "raw"
        self.files = [Path(f) for f in files] if files is not None else list_files(root, stream, start, end)
        self.stats: Counter = Counter()

    def __iter__(self) -> Iterator[StreamRecord]:
        seq = 0
        last_recv: int | None = None
        for path in self.files:
            line_no = 0
            last_in_file = 0
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    for line in fh:
                        rec = loads(line)
                        line_no += 1  # counts lines parsed, so a marker's line_no + 1 is the first bad line
                        kind = KIND_META if "meta" in rec else KIND_MSG
                        recv_ns, payload = rec["recv_ns"], rec[kind]
                        last_in_file = recv_ns
                        if kind == KIND_MSG:
                            if last_recv is not None and recv_ns - last_recv > self.gap_ns:
                                self.stats[KIND_GAP] += 1
                                info = {"gap_ns": recv_ns - last_recv, "prev_recv_ns": last_recv}
                                yield StreamRecord(path, line_no, seq, recv_ns, KIND_GAP, info)
                                seq += 1
                            last_recv = recv_ns
                            payload = self._filter(payload)
                            if payload is None:
                                continue
                        self.stats[kind] += 1
                        yield StreamRecord(path, line_no, seq, recv_ns, kind, payload)
                        seq += 1
            except _CORRUPT_TAIL as exc:
                self.stats[KIND_TRUNCATED] += 1
                info = {"error": f"{type(exc).__name__}: {exc}", "lines_read": line_no}
                yield StreamRecord(path, line_no + 1, seq, last_in_file, KIND_TRUNCATED, info)
                seq += 1

    def _filter(self, payload: dict) -> dict | None:
        """Keep only configured symbols' items; None if a data message had items but none match."""
        data = payload.get("data")
        if self.symbols is None or not isinstance(data, list):
            return payload
        kept = [it for it in data if not isinstance(it, dict) or "symbol" not in it or it["symbol"] in self.symbols]
        if data and not kept:
            return None
        return payload if len(kept) == len(data) else {**payload, "data": kept}


def book_items(record: StreamRecord, channel: str = "level3") -> Iterator[tuple[str, dict]]:
    """(msg_type, item) for each snapshot/update item of `channel` in a msg record."""
    if record.kind != KIND_MSG:
        return
    msg = record.payload
    if msg.get("channel") == channel and msg.get("type") in ("snapshot", "update"):
        for item in msg["data"]:
            yield msg["type"], item
