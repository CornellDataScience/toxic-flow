"""Stream the public L2 book (+ trades), validating every checksum. No API key needed.

    python scripts/stream_l2_book.py --symbol BTC/USD --depth 25 --seconds 60

Prints top-of-book once per --every seconds and a trade line per trade
(--trades). Ends with checksum stats; checksum_fail should be 0 or very close.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.book import L2Book
from toxicflow.kraken.ws import L2_DEPTHS, PUBLIC_WS, BookStream, fetch_pair_info


async def main(args) -> None:
    pairs = await fetch_pair_info(args.symbols)
    books = {s: L2Book(pairs[s], args.depth) for s in args.symbols}
    last_print = {s: 0.0 for s in args.symbols}

    def on_book(book: L2Book, msg_type, item, ok) -> None:
        now = time.time()
        if msg_type == "snapshot" or now - last_print[book.symbol] >= args.every:
            last_print[book.symbol] = now
            bid, ask = book.best_bid(), book.best_ask()
            print(
                f"{book.timestamp or msg_type:>30} {book.symbol:>9} "
                f"bid {book.level_qty('bids', bid):>14} @ {bid}  |  ask {ask} @ {book.level_qty('asks', ask):<14} "
                f"spread={ask - bid} levels={len(book.bids)}/{len(book.asks)} checksum={'ok' if ok else ok}"
            )

    def on_message(msg) -> None:
        if args.trades and msg.get("channel") == "trade" and msg.get("type") == "update":
            for t in msg["data"]:
                print(f"{t['timestamp']:>30} {t['symbol']:>9} TRADE {t['side']:>4} {t['qty']} @ {t['price']} ({t['ord_type']})")

    stream = BookStream(
        PUBLIC_WS,
        "book",
        books,
        extra_subs=[{"channel": "trade", "symbol": args.symbols, "snapshot": False}],
        on_book=on_book,
        on_message=on_message,
    )
    try:
        await stream.run(args.seconds)
    finally:
        print("\n" + stream.stats.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", dest="symbols", action="append", help="repeatable; default BTC/USD")
    parser.add_argument("--depth", type=int, default=25, choices=L2_DEPTHS)
    parser.add_argument("--seconds", type=float, default=30, help="run time; 0 = forever")
    parser.add_argument("--every", type=float, default=1.0, help="seconds between top-of-book prints")
    parser.add_argument("--trades", action="store_true", help="print each trade")
    args = parser.parse_args()
    args.symbols = args.symbols or ["BTC/USD"]
    args.seconds = args.seconds or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
