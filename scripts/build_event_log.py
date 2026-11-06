"""Build the event log: every L3 order event and every trade, matched to each other.

    python scripts/build_event_log.py                                   # every file under data/raw/l3/
    python scripts/build_event_log.py data/raw/l3/2026-10-18/*.jsonl.gz --match-window-ms 5

Outputs (default under data/derived/):
  l3_events.csv.gz      one row per L3 order event (snapshot orders included), with
                        book state around it and a label: new / reentry / refill /
                        fill / partial_fill / cancel / amend / unknown / snapshot
  trades.csv.gz         one row per trade, with the L3 event_ids it was matched to
  l3_valid_intervals.csv  same table as scripts/valid_intervals.py

Trades come from the public recordings for the same hours (plus one hour on each
side), unless --public is given. Without trades, reductions at the touch are
labeled unknown (no_trade_coverage). See toxicflow/kraken/eventlog.py for the rules.

Calibrate the match window from the summary: if most matched trades have dt = 0,
rerun with a small --match-window-ms (fewer false matches on a busy touch).
"""

import argparse
import csv
import gzip
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.config import DATA_DIR
from toxicflow.kraken.eventlog import EVENT_COLUMNS, TRADE_COLUMNS, EventLogBuilder, TradeIndex
from toxicflow.kraken.recorder import read_recording
from valid_intervals import write_csv as write_intervals_csv

HOUR = re.compile(r"_(\d{4}-\d{2}-\d{2}T\d{2})\.jsonl\.gz$")


def public_files_for(l3_files: list[Path], raw: Path) -> list[Path]:
    hours = set()
    for p in l3_files:
        m = HOUR.search(p.name)
        if m:
            h = datetime.strptime(m.group(1), "%Y-%m-%dT%H")
            hours.update(h + timedelta(hours=d) for d in (-1, 0, 1))
    out = [raw / "public" / h.strftime("%Y-%m-%d") / f"public_{h:%Y-%m-%dT%H}.jsonl.gz" for h in sorted(hours)]
    return [p for p in out if p.exists()]


def pct(values, p):
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))] if s else None


def summary(counts: dict, fills_behind: Counter, builder: EventLogBuilder, trades: TradeIndex, trade_rows: list) -> None:
    print("\n=== Event log summary (valid rows only) ===")
    for sym in sorted(counts):
        c = counts[sym]
        total = sum(c.values())
        valid = sum(n for (v, *_), n in c.items() if v)
        print(f"\n{sym}: {total:,} rows, {valid:,} valid ({100 * valid / total if total else 0:.2f}%)")
        print(f"  {'event_type':<10} {'action':<13} {'evidence':<18} {'rows':>12}")
        for (v, et, act, evd), n in sorted(c.items(), key=lambda kv: (kv[0][1], -kv[1])):
            if v:
                print(f"  {et:<10} {act:<13} {evd:<18} {n:>12,}")
        dts = builder.match_dts.get(sym, [])
        if dts:
            ms = [abs(d) / 1e6 for d in dts]
            print(f"  matched trades: {len(dts):,}; |trade ts - event ts| ms: exact(0) "
                  f"{100 * sum(d == 0 for d in dts) / len(dts):.1f}%, p50 {pct(ms, .5):.3f}, "
                  f"p90 {pct(ms, .9):.3f}, p99 {pct(ms, .99):.3f}, max {max(ms):.3f}")
        fills = fills_behind[sym]
        if fills:
            print(f"  fills/partial fills not at the front of the queue: {fills:,} (FIFO check; should be ~0)")
        tr = [r for r in trade_rows if r[1] == sym and r[-1] == 1]
        if tr:
            vol = sum(float(r[7]) for r in tr)
            matched = sum(float(r[9]) for r in tr)
            unmatched_n = sum(1 for r in tr if float(r[9]) < float(r[7]))
            print(f"  trades during valid L3 time: {len(tr):,}; volume matched to L3 {100 * matched / vol:.2f}%; "
                  f"{unmatched_n:,} trades not fully matched (hidden/iceberg size, timing, or outside the window)")
    print(f"\ntrade feed: {len(trades.trades):,} trades, {trades.recv_holes} receive gap(s), "
          f"{trades.id_jumps} trade_id jump(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="L3 .jsonl.gz files (default: all under data/raw/l3/)")
    parser.add_argument("--public", nargs="*", type=Path, help="public .jsonl.gz files with trades (default: matching hours)")
    parser.add_argument("--gap", type=float, default=5.0, help="silence longer than this (s) breaks validity (default 5)")
    parser.add_argument("--match-window-ms", type=float, default=100, help="max |trade ts - L3 event ts| (default 100)")
    parser.add_argument("--reentry-age-ms", type=float, default=100,
                        help="an add whose own timestamp is this much older than its message is a reentry (default 100)")
    parser.add_argument("--refill-window-s", type=float, default=60,
                        help="re-adding an order ID deleted within this long counts as a refill (default 60)")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR / "derived")
    args = parser.parse_args()

    raw = DATA_DIR / "raw"
    files = sorted(args.files or (raw / "l3").glob("*/*.jsonl.gz"), key=lambda p: p.name)
    if not files:
        sys.exit(f"no L3 files found under {raw / 'l3'}")
    public = sorted(args.public if args.public is not None else public_files_for(files, raw), key=lambda p: p.name)
    print(f"loading trades from {len(public)} public file(s)", file=sys.stderr, flush=True)
    if not public:
        print("WARNING: no public files; touch reductions will be labeled unknown", file=sys.stderr)
    trades = TradeIndex.from_files(public, gap_s=args.gap)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.out_dir / "l3_events.csv.gz"
    counts: dict[str, Counter] = {}
    fills_behind: Counter = Counter()
    with gzip.open(events_path, "wt", newline="", compresslevel=5) as fh:
        w = csv.writer(fh)
        w.writerow(EVENT_COLUMNS)

        def emit(row):
            w.writerow(row)
            c = counts.setdefault(row[2], Counter())
            c[(row[-1], row[7], row[19], row[20])] += 1
            if row[-1] and row[19] in ("fill", "partial_fill") and row[14] not in ("", "0"):
                fills_behind[row[2]] += 1

        builder = EventLogBuilder(trades, emit, args.gap, args.match_window_ms, args.reentry_age_ms,
                                  args.refill_window_s)
        for i, path in enumerate(files, 1):
            print(f"[{i}/{len(files)}] {path.name}", file=sys.stderr, flush=True)
            for rec in read_recording(path):
                builder.feed(*rec)
        builder.finish()

    trades_path = args.out_dir / "trades.csv.gz"
    trade_rows = list(builder.trade_rows())
    with gzip.open(trades_path, "wt", newline="", compresslevel=5) as fh:
        w = csv.writer(fh)
        w.writerow(TRADE_COLUMNS)
        w.writerows(trade_rows)
    intervals_path = args.out_dir / "l3_valid_intervals.csv"
    write_intervals_csv(intervals_path, sorted(builder.tracker.intervals, key=lambda iv: (iv.symbol, iv.start_ns)))

    summary(counts, fills_behind, builder, trades, trade_rows)
    print(f"\nwritten: {events_path} ({builder.event_id:,} rows), {trades_path} ({len(trade_rows):,} rows), {intervals_path}")


if __name__ == "__main__":
    main()
