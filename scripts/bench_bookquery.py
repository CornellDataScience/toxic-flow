"""Time BookQuery on recorded data.

    python scripts/bench_bookquery.py --symbol BTC/USD
    python scripts/bench_bookquery.py --l3 'data/raw/l3/2026-10-16/*.jsonl.gz' --checkpoint-every 500

Reports build time, then 10k random mid_at queries (vectorized and one at a
time), mid_after, 1k random book_at queries, and a full iter_book scan.
"""

import argparse
import glob
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.bookquery import BookQuery, build_tables


def timed(label: str, fn, n: int | None = None):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    per = f"  ({dt / n * 1e6:.2f} us/query)" if n else ""
    print(f"  {label:<36} {dt * 1e3:9.1f} ms{per}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--l3", default="data/raw/l3/*/*.jsonl.gz")
    parser.add_argument("--public", default="data/raw/public/*/*.jsonl.gz")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    args = parser.parse_args()

    tables = timed("build_tables (raw replay + checksums)",
                   lambda: build_tables(glob.glob(args.l3), glob.glob(args.public), symbol=args.symbol))
    ev = tables.events
    span = (tables.valid_intervals["end_exchange"] - tables.valid_intervals["start_exchange"]).sum() / 1e9
    print(f"  {len(ev)} events, {ev['msg_id'].nunique()} messages, {len(tables.trades)} trades, "
          f"{len(tables.valid_intervals)} valid interval(s) covering {span:.0f}s")

    with tempfile.TemporaryDirectory() as cache:
        load = lambda: BookQuery.load(ev, tables.trades, tables.valid_intervals, pair=tables.pair,  # noqa: E731
                                      depth=tables.depth, checkpoint_every=args.checkpoint_every, cache_dir=cache)
        bq = timed("BookQuery.load (build + write cache)", load)
        timed("BookQuery.load (from cache)", load)

    rng = np.random.default_rng(0)
    lo, hi = bq.valid_intervals[0][0], bq.valid_intervals[-1][1]
    t10k = rng.integers(lo, hi, 10_000)
    t1k = rng.integers(lo, hi, 1_000)
    print(f"queries (checkpoint_every={args.checkpoint_every}):")
    timed("mid_at x10k, one array call", lambda: bq.mid_at(t10k), 10_000)
    timed("mid_at x10k, scalar loop", lambda: [bq.mid_at(int(t)) for t in t10k], 10_000)
    timed("mid_after x10k (3 horizons)", lambda: bq.mid_after(t10k), 10_000)
    timed("best_at x10k, one array call", lambda: bq.best_at(t10k), 10_000)
    timed("book_at x1k", lambda: [bq.book_at(int(t)) for t in t1k], 1_000)
    timed("queue_ahead x1k (new order at best bid)",
          lambda: [bq.queue_ahead("bid", b.bid_price, int(t)) for t in t1k if (b := bq.best_at(int(t)))], 1_000)
    steps = timed("iter_book over everything", lambda: sum(1 for _ in bq.iter_book(lo, hi)))
    print(f"  ({steps} steps)")


if __name__ == "__main__":
    main()
