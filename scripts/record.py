"""Record raw Kraken feeds to data/raw/ for later replay. This is the collector.

    python scripts/record.py --symbol BTC/USD --symbol ETH/USD            # L2 + trades + L3
    python scripts/record.py --no-l3 --seconds 300                        # public feeds only

Two streams, one file per stream per UTC hour:
  public: `book` (L2, --depth) and `trade` from wss://ws.kraken.com/v2
  l3:     `level3` (--l3-depth) from wss://ws-l3.kraken.com/v2 (needs API key)

Books are maintained live so checksum failures trigger an immediate resync,
and every hourly file starts with a fresh snapshot so it can be replayed on its
own. Check a recording with scripts/validate_recording.py.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.book import L2Book, L3Book
from toxicflow.kraken.config import DATA_DIR, get_credentials
from toxicflow.kraken.recorder import Recorder
from toxicflow.kraken.rest import KrakenREST
from toxicflow.kraken.ws import L2_DEPTHS, L3_DEPTHS, L3_WS, PUBLIC_WS, BookStream, fetch_pair_info

log = logging.getLogger("record")


def meta_for(stream: str, depth: int, pairs: dict) -> dict:
    return {
        "stream": stream,
        "depth": depth,
        "pairs": {s: {"price_precision": p.price_precision, "qty_precision": p.qty_precision} for s, p in pairs.items()},
    }


async def rotate_files(jobs: list[tuple[Recorder, BookStream]]) -> None:
    while True:
        for rec, stream in jobs:
            if rec.maybe_rotate():
                log.info("rotated %s -> %s", rec.stream, rec.path)
                await stream.resync()
        await asyncio.sleep(1)


async def report_stats(jobs: list[tuple[Recorder, BookStream]], every: float) -> None:
    while True:
        await asyncio.sleep(every)
        for rec, stream in jobs:
            log.info("%s: %s -> %s", rec.stream, stream.stats.summary(), rec.path)


async def main(args) -> None:
    pairs = await fetch_pair_info(args.symbols)
    root = args.out or DATA_DIR / "raw"
    jobs = []

    rec = Recorder(root, "public", meta_for("public", args.depth, pairs))
    stream = BookStream(
        PUBLIC_WS,
        "book",
        {s: L2Book(pairs[s], args.depth) for s in args.symbols},
        extra_subs=[{"channel": "trade", "symbol": args.symbols, "snapshot": False}],
        on_raw=rec.write,
    )
    jobs.append((rec, stream))

    if not args.no_l3:
        rest = KrakenREST(*get_credentials())
        rec = Recorder(root, "l3", meta_for("l3", args.l3_depth, pairs))
        stream = BookStream(
            L3_WS,
            "level3",
            {s: L3Book(pairs[s], args.l3_depth) for s in args.symbols},
            token_provider=rest.get_ws_token,
            on_raw=rec.write,
        )
        jobs.append((rec, stream))

    for rec, _ in jobs:
        rec.maybe_rotate()
    tasks = [asyncio.create_task(rotate_files(jobs)), asyncio.create_task(report_stats(jobs, args.stats_every))]
    try:
        await asyncio.gather(*(stream.run(args.seconds) for _, stream in jobs))
    finally:
        for t in tasks:
            t.cancel()
        for rec, stream in jobs:
            rec.close()
            log.info("%s final: %s", rec.stream, stream.stats.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", dest="symbols", action="append", help="repeatable; default BTC/USD")
    parser.add_argument("--depth", type=int, default=100, choices=L2_DEPTHS, help="L2 depth")
    parser.add_argument("--l3-depth", type=int, default=100, choices=L3_DEPTHS)
    parser.add_argument("--no-l3", action="store_true", help="skip the authenticated L3 feed")
    parser.add_argument("--seconds", type=float, default=0, help="run time; 0 = forever")
    parser.add_argument("--stats-every", type=float, default=60)
    parser.add_argument("--out", help="output root (default: data/raw)")
    args = parser.parse_args()
    args.symbols = args.symbols or ["BTC/USD"]
    args.seconds = args.seconds or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
