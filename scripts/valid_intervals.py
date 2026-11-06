"""Find the time intervals where recorded L3 data is trustworthy, and report data quality.

    python scripts/valid_intervals.py                                  # every file under data/raw/l3/
    python scripts/valid_intervals.py data/raw/l3/2026-10-18/*.jsonl.gz --gap 5 --min-duration 60

Replays L3 recordings (in file-name = time order) through the local book. A
symbol's data counts as valid from a snapshot with a matching checksum until
the first problem: a checksum failure, a silence longer than --gap seconds, a
reconnect, a file boundary, or the end of the data. See
toxicflow/kraken/intervals.py for the exact rules.

Outputs:
  - a table of valid intervals per symbol (also written as CSV, see --csv)
  - a summary: checksum pass rate, desync periods, longest message gap, and
    the share of recorded time that is valid
"""

import argparse
import csv
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.config import DATA_DIR
from toxicflow.kraken.intervals import IntervalTracker
from toxicflow.kraken.recorder import read_recording


def utc(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def hms(seconds: float) -> str:
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}h{int(m):02d}m{s:04.1f}s" if h else f"{int(m)}m{s:04.1f}s" if m else f"{s:.1f}s"


def print_table(intervals) -> None:
    header = f"{'symbol':<9} {'start (UTC)':<23} {'end (UTC)':<23} {'duration':>12} {'start_reason':<16} {'end_reason':<14} {'updates':>8}"
    print(header)
    print("-" * len(header))
    for iv in intervals:
        print(
            f"{iv.symbol:<9} {utc(iv.start_ns):<23} {utc(iv.end_ns):<23} {hms(iv.duration_s):>12} "
            f"{iv.start_reason:<16} {iv.end_reason:<14} {iv.updates:>8}"
        )


def write_csv(path: Path, intervals) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["symbol", "start_utc", "end_utc", "start_ns", "end_ns", "duration_s", "start_reason", "end_reason", "updates"])
        for iv in intervals:
            w.writerow([iv.symbol, utc(iv.start_ns), utc(iv.end_ns), iv.start_ns, iv.end_ns,
                        f"{iv.duration_s:.3f}", iv.start_reason, iv.end_reason, iv.updates])


def print_summary(t: IntervalTracker, intervals, args) -> None:
    span = t.span_s
    print("\n=== Summary ===")
    print(f"recording:        {utc(t.first_ns)} -> {utc(t.last_ns)} UTC ({hms(span)}), "
          f"{len(args.files)} file(s), {t.sessions} writer session(s), {t.reconnects} reconnect(s)")
    print(f"longest msg gap:  {t.longest_gap_s:.2f}s (starting {utc(t.longest_gap_at_ns)} UTC)" if t.longest_gap_at_ns else
          "longest msg gap:  n/a")
    print(f"gaps > {args.gap:g}s:      {len(t.gaps)} totalling {sum(g for _, g in t.gaps):.1f}s")
    for start, g in sorted(t.gaps, key=lambda x: -x[1])[:10]:
        print(f"    {utc(start)} UTC  {g:.1f}s")
    if args.min_duration:
        print(f"min interval:     {args.min_duration:g}s (shorter valid intervals are dropped and count as invalid)")

    total_valid = 0.0
    total_ok = total_fail = 0
    for sym, st in sorted(t.stats.items()):
        ivs = [iv for iv in intervals if iv.symbol == sym]
        valid = sum(iv.duration_s for iv in ivs)
        total_valid += valid
        total_ok += st.checksum_ok
        total_fail += st.checksum_fail
        rate = st.checksum_pass_rate
        ends = Counter(iv.end_reason for iv in ivs)
        print(f"\n{sym}")
        print(f"  checksum pass rate     {100 * rate:.4f}%  ({st.checksum_ok:,} ok / {st.checksum_fail:,} fail)" if rate is not None
              else "  checksum pass rate     n/a")
        print(f"  desync periods         {st.desync_periods}"
              + (f"  (total {sum(st.desync_s):.1f}s, longest {max(st.desync_s):.1f}s, median {sorted(st.desync_s)[len(st.desync_s) // 2]:.2f}s)"
                 if st.desync_s else ""))
        print(f"  resync breaks          {ends['resnapshot']}  (fresh snapshot mid-interval: the recorder resynced, "
              f"so there's a short hole even though our book still matched)")
        print(f"  snapshots / updates    {st.snapshots:,} / {st.updates:,}  (skipped while unsynced: {st.skipped_unsynced:,})")
        print(f"  valid intervals        {len(ivs)}  (longest {hms(max((iv.duration_s for iv in ivs), default=0))}, "
              f"median {hms(sorted(iv.duration_s for iv in ivs)[len(ivs) // 2] if ivs else 0)})")
        print(f"  interval end reasons   {dict(ends.most_common())}")
        if st.longest_update_gap_at_ns:
            print(f"  longest update gap     {st.longest_update_gap_s:.2f}s inside a valid interval (starting {utc(st.longest_update_gap_at_ns)} UTC)")
        print(f"  valid time             {hms(valid)} of {hms(span)} = {100 * valid / span if span else 0:.2f}%")

    n = len(t.stats)
    checked = total_ok + total_fail
    print(f"\nOVERALL ({n} symbol(s))")
    print(f"  checksum pass rate     {100 * total_ok / checked if checked else 0:.4f}%")
    print(f"  desync periods         {sum(st.desync_periods for st in t.stats.values())}")
    print(f"  resync breaks          {sum(1 for iv in intervals if iv.end_reason == 'resnapshot')}")
    print(f"  longest message gap    {t.longest_gap_s:.2f}s")
    print(f"  valid data             {100 * total_valid / (span * n) if span and n else 0:.2f}% of symbol-time")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="L3 .jsonl.gz files (default: all under data/raw/l3/)")
    parser.add_argument("--gap", type=float, default=5.0, help="silence longer than this (s) breaks validity (default 5)")
    parser.add_argument("--min-duration", type=float, default=0.0, help="drop valid intervals shorter than this (s)")
    parser.add_argument("--csv", type=Path, default=DATA_DIR / "derived" / "l3_valid_intervals.csv", help="interval table output")
    parser.add_argument("--no-table", action="store_true", help="print only the summary")
    args = parser.parse_args()

    args.files = sorted(args.files or (DATA_DIR / "raw" / "l3").glob("*/*.jsonl.gz"))
    if not args.files:
        sys.exit(f"no L3 files found under {DATA_DIR / 'raw' / 'l3'}")

    tracker = IntervalTracker(args.gap)
    for i, path in enumerate(args.files, 1):
        print(f"[{i}/{len(args.files)}] {path.name}", file=sys.stderr, flush=True)
        for rec in read_recording(path):
            tracker.feed(*rec)
    tracker.finish()

    intervals = sorted((iv for iv in tracker.intervals if iv.duration_s >= args.min_duration),
                       key=lambda iv: (iv.symbol, iv.start_ns))
    if not args.no_table:
        print_table(intervals)
    write_csv(args.csv, intervals)
    print_summary(tracker, intervals, args)
    print(f"\ninterval table written to {args.csv}")


if __name__ == "__main__":
    main()
