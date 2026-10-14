"""Raw message archive: one gzipped JSONL file per stream per UTC hour.

Layout: <root>/<stream>/<YYYY-MM-DD>/<stream>_<YYYY-MM-DDTHH>.jsonl.gz

Each file starts with a meta line, followed by one line per websocket message:
    {"recv_ns": 1792010262000000000, "meta": {...}}
    {"recv_ns": 1792010262123456789, "msg": <exact JSON text Kraken sent>}

The message text is spliced in verbatim so no numeric precision is lost.
Restarting within the same hour appends a new gzip member (with its own meta
line) to the existing file; gzip readers handle this transparently.
"""

import gzip
import json
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from .ws import loads


class Recorder:
    def __init__(self, root: Path, stream: str, meta: dict):
        self.root = Path(root)
        self.stream = stream
        self.meta = meta
        self.path: Path | None = None
        self._fh = None
        self._hour: str | None = None

    def maybe_rotate(self) -> bool:
        """Open the file for the current hour. True if an existing file was rotated out."""
        hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
        if hour == self._hour:
            return False
        rotated = self._fh is not None
        self.close()
        self.path = self.root / self.stream / hour[:10] / f"{self.stream}_{hour}.jsonl.gz"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = gzip.open(self.path, "at", encoding="utf-8")
        self._hour = hour
        self._fh.write(json.dumps({"recv_ns": time.time_ns(), "meta": self.meta}) + "\n")
        return rotated

    def write(self, recv_ns: int, raw: str, msg: dict) -> None:
        if self._fh is None:
            self.maybe_rotate()
        self._fh.write(f'{{"recv_ns":{recv_ns},"msg":{raw}}}\n')

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_recording(path: Path) -> Iterator[tuple[int, str, dict]]:
    """Yield (recv_ns, kind, payload) where kind is "meta" or "msg".

    A file whose writer was killed ends in a truncated gzip member / partial
    line; reading stops there instead of raising.
    """
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        try:
            for line in fh:
                rec = loads(line)
                kind = "meta" if "meta" in rec else "msg"
                yield rec["recv_ns"], kind, rec[kind]
        except (EOFError, json.JSONDecodeError):
            return
