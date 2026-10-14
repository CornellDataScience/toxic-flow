"""Replay recorded files, rebuild the books offline, and report data quality.

    python scripts/validate_recording.py data/raw/public/2026-10-14/*.jsonl.gz
    python scripts/validate_recording.py data/raw/l3/**/*.jsonl.gz

Files are replayed in name (= time) order. Reports per symbol: snapshots,
updates, checksum failures (the <0.1% desync OKR), trades, and any silent gaps
in the stream longer than --gap seconds (disconnects / downtime).
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.book import L2Book, L3Book, PairInfo
from toxicflow.kraken.recorder import read_recording

BOOK_TYPES = {"book": L2Book, "level3": L3Book}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--gap", type=float, default=5.0, help="report receive gaps longer than this (s)")
    args = parser.parse_args()

    books: dict[tuple[str, str], L2Book | L3Book] = {}
    meta = None
    counts = defaultdict(lambda: defaultdict(int))
    gaps: list[tuple[int, float]] = []
    first_ns = last_ns = None

    for path in sorted(args.files):
        for recv_ns, kind, payload in read_recording(path):
            if kind == "meta":
                meta = payload
                books.clear()  # a new writer session; wait for fresh snapshots
                continue
            if last_ns is not None and (recv_ns - last_ns) / 1e9 > args.gap:
                gaps.append((last_ns, (recv_ns - last_ns) / 1e9))
            first_ns = first_ns or recv_ns
            last_ns = recv_ns

            channel, msg_type = payload.get("channel"), payload.get("type")
            if channel == "trade" and msg_type == "update":
                for t in payload["data"]:
                    counts[t["symbol"]]["trades"] += 1
            elif channel in BOOK_TYPES and msg_type in ("snapshot", "update"):
                for item in payload["data"]:
                    sym = item["symbol"]
                    book = books.get((channel, sym))
                    if book is None:
                        p = meta["pairs"][sym]
                        book = BOOK_TYPES[channel](PairInfo(sym, p["price_precision"], p["qty_precision"]), meta["depth"])
                        books[(channel, sym)] = book
                    ok = book.apply(msg_type, item)
                    c = counts[sym]
                    c[f"{channel}_{msg_type}s"] += 1
                    c[{True: "checksum_ok", False: "checksum_fail", None: "skipped_unsynced"}[ok]] += 1

    if first_ns is None:
        print("no messages found")
        return
    hours = (last_ns - first_ns) / 3.6e12
    print(f"span: {hours:.2f}h across {len(args.files)} file(s)")
    for sym, c in sorted(counts.items()):
        checked = c["checksum_ok"] + c["checksum_fail"]
        fail_pct = 100 * c["checksum_fail"] / checked if checked else 0.0
        print(f"\n{sym}")
        for k, v in sorted(c.items()):
            print(f"  {k:<22} {v}")
        print(f"  {'checksum_fail_rate':<22} {fail_pct:.4f}%")
    downtime = sum(g for _, g in gaps)
    print(f"\ngaps > {args.gap}s: {len(gaps)} totalling {downtime:.1f}s "
          f"(uptime {100 * (1 - downtime / (hours * 3600)) if hours else 100:.2f}%)")
    for start_ns, g in gaps[:20]:
        print(f"  at recv_ns={start_ns}: {g:.1f}s")


if __name__ == "__main__":
    main()
