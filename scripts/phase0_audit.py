"""Phase 0 audit of L3 recordings: discontinuity spacing, horizon exposure, touch depth, modify mix.

    python scripts/phase0_audit.py                                   # every file under data/raw/l3/
    python scripts/phase0_audit.py data/raw/l3/2026-10-18/*.jsonl.gz --list

Read-only; prints markdown to stdout. The clock is the item (matching-engine)
timestamp, as in the design (§7.1 R2). Per symbol, a synced interval runs from a
snapshot whose checksum passes until one of these discontinuities:

    checksum    an update fails its checksum
    resnapshot  a snapshot arrives while the book is synced (resync / reconnect)
    file        the next file's meta line (every hourly file restarts the book)
    gap         the reader's receive-gap marker (> --gap seconds of silence)
    truncated   the reader's truncated-file marker

A file boundary while the book is already unsynced is not counted again. The
last interval ends at the end of the data, which censors rather than interrupts.
Touch statistics are sampled once per second over synced time, so they are
time-weighted. Modify events are classified only from items whose checksum passed.
"""

import argparse
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.execsim.reader import KIND_GAP, KIND_META, KIND_TRUNCATED, RecordingReader, book_items, list_files
from toxicflow.execsim.sim_book import SimL3Book
from toxicflow.kraken.book import PairInfo
from toxicflow.kraken.config import DATA_DIR

S = 1_000_000_000
HORIZONS_S = (10, 30, 60, 300)


def parse_ts_ns(ts: str) -> int:
    """'2026-10-18T02:06:55.103101894Z' -> integer ns since the epoch, without float rounding."""
    base, _, frac = ts.rstrip("Z").partition(".")
    secs = int(datetime.strptime(base, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp())
    return secs * S + int(frac.ljust(9, "0")[:9])


def pct(values: list, p: float):
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)]


@dataclass
class SymbolAudit:
    counts: Counter = field(default_factory=Counter)
    precisions: set = field(default_factory=set)  # (price_precision, qty_precision) seen in meta lines
    intervals: list = field(default_factory=list)  # (start_ns, end_ns, cause)
    discontinuities: list = field(default_factory=list)  # (ts_ns, cause)
    modifies: Counter = field(default_factory=Counter)
    touch: dict = field(default_factory=lambda: {"bid_qty": [], "ask_qty": [], "bid_n": [], "ask_n": []})
    open_start: int | None = None
    last_ts: int | None = None
    next_sample: int | None = None

    def open(self, ts: int) -> None:
        self.open_start = self.next_sample = self.last_ts = ts

    def close(self, cause: str, ts: int | None = None) -> None:
        if self.open_start is None:
            return
        end = ts if ts is not None else self.last_ts
        self.intervals.append((self.open_start, end, cause))
        if cause != "end":
            self.discontinuities.append((end, cause))
        self.open_start = self.next_sample = None

    def sample_until(self, ts: int, book: SimL3Book) -> None:
        """Record the touch at each whole second since the last sample, using the state before this item."""
        if self.open_start is None:
            return
        while self.next_sample < ts:
            for side, prefix in (("bids", "bid"), ("asks", "ask")):
                top = book.levels(side, 1)
                if top:
                    _, qty, n_orders = top[0]
                    self.touch[f"{prefix}_qty"].append(qty)
                    self.touch[f"{prefix}_n"].append(n_orders)
            self.next_sample += S


def classify_modify(ev) -> str:
    if ev.qty_before is None:
        return "unknown_id"
    if ev.price != ev.price_before:
        return "price_change"
    if ev.qty > ev.qty_before:
        return "qty_increase"
    if ev.qty < ev.qty_before:
        return "qty_decrease"
    return "unchanged"


def horizon_share(intervals: list, horizon_ns: int) -> tuple[float | None, int]:
    """Share of 1 s-spaced start times inside synced intervals that meet a discontinuity before expiry.

    A start t is hit if the interval ends at e with t + H > e (an event at exactly t + H comes after
    expiry, §7.1 R6). Starts in the final, censored interval count only if t + H fits before the data ends.
    """
    hits = total = 0
    for start, end, cause in intervals:
        length = end - start
        n_starts = -(-length // S)  # k = 0 .. n-1 with k*S < length
        fits = (length - horizon_ns) // S + 1 if length >= horizon_ns else 0  # k with k*S + H <= length
        if cause == "end":
            total += min(n_starts, fits)
        else:
            total += n_starts
            hits += n_starts - min(n_starts, fits)
    return (hits / total if total else None), total


def run(files: list[Path], gap_s: float, symbols: set[str] | None):
    audits: dict[str, SymbolAudit] = {}
    per_file: dict[tuple[str, str], Counter] = {}
    books: dict[str, SimL3Book] = {}
    meta = None
    first_ts = None
    for rec in RecordingReader(files=files, gap_ns=int(gap_s * S), symbols=symbols):
        if rec.kind == KIND_META:
            meta = rec.payload
            for sym, a in audits.items():
                a.close("file")
            for sym, p in meta["pairs"].items():
                audits.setdefault(sym, SymbolAudit()).precisions.add((p["price_precision"], p["qty_precision"]))
            books.clear()
            continue
        if rec.kind in (KIND_GAP, KIND_TRUNCATED):
            for a in audits.values():
                a.close(rec.kind)
            continue
        for msg_type, item in book_items(rec):
            sym = item["symbol"]
            if symbols and sym not in symbols:
                continue
            book = books.get(sym)
            if book is None:
                p = meta["pairs"][sym]
                pair = PairInfo(sym, p["price_precision"], p["qty_precision"])
                book = books[sym] = SimL3Book(pair, meta["depth"], track_queue=False)
            a = audits.setdefault(sym, SymbolAudit())
            fc = per_file.setdefault((rec.file.name, sym), Counter())
            ts = parse_ts_ns(item["timestamp"])
            first_ts = ts if first_ts is None else first_ts

            a.sample_until(ts, book)
            if msg_type == "snapshot":
                a.close("resnapshot", ts)
            ok = book.apply(msg_type, item)
            key = {True: "ok", False: "checksum_fail", None: "unsynced"}[ok]
            for c in (a.counts, fc):
                c[msg_type] += 1
                c[key] += 1
                c[f"{msg_type}_{key}"] += 1
            if ok is True:
                if a.open_start is None:
                    a.open(ts)
                a.last_ts = ts
                for ev in book.last_events:
                    if ev.kind == "modify":
                        a.modifies[classify_modify(ev)] += 1
                        fc["modify"] += 1
            elif ok is False:
                a.close("checksum", ts)
    for a in audits.values():
        a.close("end")
    return audits, per_file, first_ts


def fmt_s(ns) -> str:
    return "-" if ns is None else f"{ns / S:,.1f}"


def fmt_q(q, places: int) -> str:
    return "-" if q is None else f"{q:.{places}f}"


def report(audits, per_file, first_ts, files, list_times: bool) -> None:
    print(f"# Phase 0 L3 audit\n\n{len(files)} file(s): {files[0].name} .. {files[-1].name}\n")

    print("## Per file\n")
    print("| file | symbol | snapshots | updates OK | checksum failures | unsynced updates | modifies |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for (name, sym), c in per_file.items():
        print(f"| {name} | {sym} | {c['snapshot']} | {c['update_ok']} "
              f"| {c['checksum_fail']} | {c['unsynced']} | {c['modify']} |")

    print("\n## 1. Discontinuities (seconds between consecutive ones)\n")
    print("| symbol | snapshots | checksum failures | discontinuities (by cause) | min | median | p90 | max |")
    print("|---|---:|---:|---|---:|---:|---:|---:|")
    for sym, a in audits.items():
        times = [t for t, _ in a.discontinuities]
        spacing = [b - t for t, b in zip(times, times[1:])]
        causes = ", ".join(f"{k} {v}" for k, v in Counter(c for _, c in a.discontinuities).most_common())
        print(f"| {sym} | {a.counts['snapshot']} | {a.counts['checksum_fail']} | {len(times)} ({causes}) "
              f"| {fmt_s(pct(spacing, 0))} | {fmt_s(pct(spacing, 0.5))} | {fmt_s(pct(spacing, 0.9))} "
              f"| {fmt_s(pct(spacing, 1))} |")
    if list_times:
        for sym, a in audits.items():
            print(f"\n{sym} discontinuities (s from first record): "
                  + ", ".join(f"{(t - first_ts) / S:.1f} {c}" for t, c in a.discontinuities))

    print("\n## 2. Share of 1 s start times that meet a discontinuity within the horizon\n")
    print("| symbol | synced time (h) | " + " | ".join(f"{h} s" for h in HORIZONS_S) + " |")
    print("|---|---:|" + "---:|" * len(HORIZONS_S))
    for sym, a in audits.items():
        synced_h = sum(e - s for s, e, _ in a.intervals) / S / 3600
        cells = []
        for h in HORIZONS_S:
            share, n = horizon_share(a.intervals, h * S)
            cells.append("-" if share is None else f"{100 * share:.2f}% (n={n:,})")
        print(f"| {sym} | {synced_h:.2f} | " + " | ".join(cells) + " |")

    print("\n## 3. Touch depth over synced time (base units, sampled every 1 s)\n")
    print("| symbol | side | p10 qty | median qty | p90 qty | median orders at best | samples |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for sym, a in audits.items():
        qp = max(q for _, q in a.precisions) if a.precisions else 8
        for prefix, side in (("bid", "bid"), ("ask", "ask")):
            qty, n = a.touch[f"{prefix}_qty"], a.touch[f"{prefix}_n"]
            print(f"| {sym} | {side} | {fmt_q(pct(qty, 0.1), qp)} | {fmt_q(pct(qty, 0.5), qp)} "
                  f"| {fmt_q(pct(qty, 0.9), qp)} | {pct(n, 0.5) if n else '-'} | {len(qty):,} |")

    print("\n## 4. Modify events (checksum-valid items only)\n")
    kinds = ("qty_decrease", "qty_increase", "price_change", "unchanged", "unknown_id")
    print("| symbol | modifies | " + " | ".join(k.replace("_", " ") for k in kinds) + " |")
    print("|---|---:|" + "---:|" * len(kinds))
    for sym, a in audits.items():
        print(f"| {sym} | {sum(a.modifies.values()):,} | " + " | ".join(f"{a.modifies[k]:,}" for k in kinds) + " |")

    print("\n## 5. Precision and tick (from meta lines; tick = 10**-price_precision)\n")
    print("| symbol | price_precision | qty_precision | tick |")
    print("|---|---:|---:|---:|")
    for sym, a in audits.items():
        for pp, qp in sorted(a.precisions):
            print(f"| {sym} | {pp} | {qp} | {Decimal(1).scaleb(-pp)} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="default: every file under data/raw/l3/")
    parser.add_argument("--symbol", dest="symbols", action="append", help="repeatable; default all")
    parser.add_argument("--gap", type=float, default=5.0, help="receive gap that counts as a discontinuity (s)")
    parser.add_argument("--list", action="store_true", help="also list every discontinuity time")
    args = parser.parse_args()
    files = sorted(args.files, key=lambda p: p.name) if args.files else list_files(DATA_DIR / "raw", "l3")
    if not files:
        sys.exit(f"no L3 recordings under {DATA_DIR / 'raw' / 'l3'}")
    symbols = set(args.symbols) if args.symbols else None
    audits, per_file, first_ts = run(files, args.gap, symbols)
    report(audits, per_file, first_ts, files, args.list)


if __name__ == "__main__":
    main()
