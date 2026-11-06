"""Build events / trades / valid_intervals tables straight from raw recordings.

This is a stand-in for the teammates' event log builder and valid_intervals
outputs, which don't exist yet. It produces tables in the shape BookQuery.load
expects, so swapping in theirs later only means passing different DataFrames.

    tables = build_tables(l3_paths, public_paths, symbol="BTC/USD")
    bq = BookQuery.load(tables.events, tables.trades, tables.valid_intervals,
                        pair=tables.pair, depth=tables.depth)

Validity is decided by replaying the L3 feed and checking Kraken's checksum
after every message. A valid interval opens at each snapshot and closes at:
- a checksum mismatch: ends 1 ns before the bad message, so no query ever
  sees its effects; stays closed until the next snapshot
- a silence longer than gap_s (by local receive time, heartbeats included):
  ends at the last book message before the silence
- a recorder restart (meta line) or end of the data: ends at the last book message
- exchange time going backwards: ends at the last good message

Every interval is computed in both clocks (exchange `timestamp` and local
`recv_ns`) so either can be used downstream.
"""

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from toxicflow.kraken.book import PairInfo
from toxicflow.kraken.recorder import read_recording

from .book import ASK, BID, EVENT_CODES, SIDE_NAMES, SNAPSHOT, Book
from .units import decimal_to_ticks, parse_iso_ns

log = logging.getLogger(__name__)

EVENT_COLUMNS = ["seq", "msg_id", "ts_exchange", "ts_recv", "symbol", "order_id", "side",
                 "price_ticks", "qty_lots", "event"]
TRADE_COLUMNS = ["trade_id", "ts_exchange", "ts_recv", "symbol", "side", "price_ticks", "qty_lots", "ord_type"]
INTERVAL_COLUMNS = ["symbol", "start_exchange", "end_exchange", "start_recv", "end_recv", "end_reason"]


@dataclass
class Tables:
    """events: one row per L3 order event while the book was in sync
        (snapshot orders included as event="snapshot"), in message order.
        `seq` is the global tie-breaker, `msg_id` groups events of one message.
    trades: one row per trade; `side` is the aggressor (taker) side.
    valid_intervals: [start, end] inclusive, in ns, for both clocks.
    """

    events: pd.DataFrame
    trades: pd.DataFrame
    valid_intervals: pd.DataFrame
    pair: PairInfo
    depth: int
    stats: Counter = field(default_factory=Counter)


@dataclass
class _Open:
    start_ex: int
    start_rv: int
    last_ex: int
    last_rv: int


def build_tables(
    l3_paths: Iterable[Path | str],
    public_paths: Iterable[Path | str] = (),
    *,
    symbol: str,
    gap_s: float = 5.0,
) -> Tables:
    """Replay L3 (and optionally public trade) recordings into tables. Files are read in name (= time) order."""
    events: list[tuple] = []
    intervals: list[tuple] = []
    stats: Counter = Counter()
    pair: PairInfo | None = None
    depth: int | None = None
    book: Book | None = None
    cur: _Open | None = None
    seq = msg_id = 0
    last_recv: int | None = None
    gap_ns = int(gap_s * 1e9)

    def close(end_ex: int, end_rv: int, reason: str) -> None:
        nonlocal cur, book
        if cur is not None and end_ex >= cur.start_ex and end_rv >= cur.start_rv:
            intervals.append((symbol, cur.start_ex, end_ex, cur.start_rv, end_rv, reason))
        cur, book = None, None

    for path in sorted(Path(p) for p in l3_paths):
        for recv, kind, payload in read_recording(path):
            if kind == "meta":
                if cur is not None:
                    close(cur.last_ex, cur.last_rv, "restart")
                info = payload["pairs"].get(symbol)
                if info is None:
                    raise ValueError(f"{symbol} not recorded in {path}")
                new_pair = PairInfo(symbol, int(info["price_precision"]), int(info["qty_precision"]))
                if pair is not None and (new_pair != pair or payload["depth"] != depth):
                    raise ValueError(f"{path}: precision/depth changed mid-dataset")
                pair, depth = new_pair, int(payload["depth"])
                last_recv = None
                continue
            if last_recv is not None and recv - last_recv > gap_ns and cur is not None:
                stats["gaps"] += 1
                close(cur.last_ex, cur.last_rv, "gap")
            last_recv = recv
            if payload.get("channel") != "level3" or payload.get("type") not in ("snapshot", "update"):
                continue
            assert pair is not None and depth is not None, "message before meta line"
            msg_type = payload["type"]
            for item in payload["data"]:
                if item["symbol"] != symbol:
                    continue
                ts = parse_iso_ns(item["timestamp"])
                if msg_type == "snapshot":
                    if cur is not None:
                        close(ts - 1, recv - 1, "resnapshot")
                    book = Book(pair, depth)
                    cur = _Open(ts, recv, ts, recv)
                    stats["snapshots"] += 1
                elif cur is None or book is None:
                    stats["skipped_unsynced"] += 1
                    continue
                elif ts < cur.last_ex:
                    stats["exchange_ts_backwards"] += 1
                    close(cur.last_ex, cur.last_rv, "ts_backwards")
                    continue
                rows = []
                for side, name in ((BID, "bids"), (ASK, "asks")):
                    for ev in item.get(name, ()):
                        code = SNAPSHOT if msg_type == "snapshot" else EVENT_CODES[ev["event"]]
                        price = decimal_to_ticks(ev["limit_price"], pair.price_precision)
                        qty = decimal_to_ticks(ev["order_qty"], pair.qty_precision)
                        book.apply(code, side, ev["order_id"], price, qty)
                        rows.append((ev["order_id"], SIDE_NAMES[side], price, qty,
                                     "snapshot" if code == SNAPSHOT else ev["event"]))
                book.truncate()
                if book.checksum() != item["checksum"]:
                    stats["checksum_fail"] += 1
                    log.warning("%s checksum mismatch at %s; invalid until next snapshot", symbol, item["timestamp"])
                    close(ts - 1, recv - 1, "checksum")
                    continue
                stats["checksum_ok"] += 1
                for oid, side_name, price, qty, ev_name in rows:
                    events.append((seq, msg_id, ts, recv, symbol, oid, side_name, price, qty, ev_name))
                    seq += 1
                msg_id += 1
                cur.last_ex, cur.last_rv = ts, recv
    if cur is not None:
        close(cur.last_ex, cur.last_rv, "end_of_data")
    if pair is None or depth is None:
        raise ValueError("no L3 recordings found")

    events_df = pd.DataFrame(events, columns=EVENT_COLUMNS).astype(
        {"seq": "int64", "msg_id": "int64", "ts_exchange": "int64", "ts_recv": "int64",
         "price_ticks": "int64", "qty_lots": "int64"})
    intervals_df = pd.DataFrame(intervals, columns=INTERVAL_COLUMNS)
    trades_df = _read_trades(public_paths, symbol, pair)
    stats["events"] = len(events_df)
    stats["trades"] = len(trades_df)
    return Tables(events_df, trades_df, intervals_df, pair, depth, stats)


def _read_trades(paths: Iterable[Path | str], symbol: str, pair: PairInfo) -> pd.DataFrame:
    rows: dict[int, tuple] = {}  # trade_id -> row; the subscribe snapshot repeats recent trades
    for path in sorted(Path(p) for p in paths):
        for recv, kind, payload in read_recording(path):
            if kind != "msg" or payload.get("channel") != "trade":
                continue
            for t in payload["data"]:
                if t["symbol"] != symbol or t["trade_id"] in rows:
                    continue
                rows[t["trade_id"]] = (
                    t["trade_id"], parse_iso_ns(t["timestamp"]), recv, symbol, t["side"],
                    decimal_to_ticks(t["price"], pair.price_precision),
                    decimal_to_ticks(t["qty"], pair.qty_precision), t.get("ord_type"),
                )
    df = pd.DataFrame(sorted(rows.values()), columns=TRADE_COLUMNS)
    return df.astype({"trade_id": "int64", "ts_exchange": "int64", "ts_recv": "int64",
                      "price_ticks": "int64", "qty_lots": "int64"})
