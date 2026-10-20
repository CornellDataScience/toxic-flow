"""Builders for synthetic L3 items and recordings (plain code: .gz fixtures are gitignored)."""

import gzip
import itertools
import json
import re
from decimal import Decimal
from pathlib import Path

from toxicflow.kraken.book import L3Book, PairInfo

BTC = PairInfo("BTC/USD", 1, 8)
ETH = PairInfo("ETH/USD", 2, 8)
META = {"stream": "l3", "depth": 10, "pairs": {"BTC/USD": {"price_precision": 1, "qty_precision": 8},
                                              "ETH/USD": {"price_precision": 2, "qty_precision": 8}}}

_clock = itertools.count(1)


def ts() -> str:
    return f"2026-10-18T02:00:00.{next(_clock):09d}Z"


def order(oid: str, price: str, qty: str, event: str | None = None) -> dict:
    o = {"order_id": oid, "limit_price": Decimal(price), "order_qty": Decimal(qty), "timestamp": ts()}
    return {"event": event, **o} if event else o


class Sealer:
    """Builds items with correct checksums by applying them to a reference L3Book."""

    def __init__(self, pair: PairInfo = BTC, depth: int = 10):
        self.pair = pair
        self.book = L3Book(pair, depth)

    def snapshot(self, bids=(), asks=()) -> dict:
        return self._seal("snapshot", bids, asks)

    def update(self, bids=(), asks=()) -> dict:
        return self._seal("update", bids, asks)

    def _seal(self, msg_type, bids, asks) -> dict:
        item = {"symbol": self.pair.symbol, "bids": list(bids), "asks": list(asks), "timestamp": ts(), "checksum": -1}
        self.book.apply(msg_type, item)
        item["checksum"] = self.book.checksum()
        self.book.synced = True
        return item


def level3(msg_type: str, *items: dict) -> dict:
    return {"channel": "level3", "type": msg_type, "data": list(items)}


def dumps(obj) -> str:
    """JSON with Decimals written as bare numbers, as Kraken sends them."""
    text = json.dumps(obj, default=lambda d: f"__DEC{d}__")
    return re.sub(r'"__DEC([^_"]*)__"', r"\1", text)


def gz_member(lines: list[str]) -> bytes:
    return gzip.compress("".join(line + "\n" for line in lines).encode())


def recording_lines(msgs: list[tuple[int, dict]], meta: dict | None = META, meta_ns: int = 0) -> list[str]:
    lines = [dumps({"recv_ns": meta_ns, "meta": meta})] if meta is not None else []
    return lines + [f'{{"recv_ns":{ns},"msg":{dumps(m)}}}' for ns, m in msgs]


def write_file(root: Path, hour: str, data: bytes, stream: str = "l3") -> Path:
    path = Path(root, stream, hour[:10], f"{stream}_{hour}.jsonl.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
