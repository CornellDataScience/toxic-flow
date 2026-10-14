"""Stream the authenticated L3 (order-by-order) book, validating every checksum.

    python scripts/stream_l3_orders.py --symbol BTC/USD --depth 10 --seconds 60

Prints individual add/modify/delete events (--events), plus a periodic book
summary. Requires KRAKEN_API_KEY/SECRET in .env.
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.kraken.book import L3Book
from toxicflow.kraken.config import get_credentials
from toxicflow.kraken.rest import KrakenREST
from toxicflow.kraken.ws import L3_DEPTHS, L3_WS, BookStream, fetch_pair_info


async def main(args) -> None:
    rest = KrakenREST(*get_credentials())
    pairs = await fetch_pair_info(args.symbols)
    books = {s: L3Book(pairs[s], args.depth) for s in args.symbols}
    last_print = {s: 0.0 for s in args.symbols}

    def on_book(book: L3Book, msg_type, item, ok) -> None:
        if args.events and msg_type == "update":
            for name in ("bids", "asks"):
                for ev in item.get(name, ()):
                    print(
                        f"{ev['timestamp']:>30} {book.symbol:>9} {name[:-1]:>3} {ev['event']:<6} "
                        f"{ev['order_id']}  {ev['order_qty']} @ {ev['limit_price']}"
                    )
        now = time.time()
        if msg_type == "snapshot" or now - last_print[book.symbol] >= args.every:
            last_print[book.symbol] = now
            print(
                f"--- {book.timestamp or msg_type} {book.symbol} bid={book.best_bid()} ask={book.best_ask()} "
                f"orders={len(book.orders)} checksum={'ok' if ok else ok} events={dict(book.event_counts)}"
            )

    stream = BookStream(L3_WS, "level3", books, token_provider=rest.get_ws_token, on_book=on_book)
    try:
        await stream.run(args.seconds)
    finally:
        print("\n" + stream.stats.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", dest="symbols", action="append", help="repeatable; default BTC/USD")
    parser.add_argument("--depth", type=int, default=10, choices=L3_DEPTHS)
    parser.add_argument("--seconds", type=float, default=30, help="run time; 0 = forever")
    parser.add_argument("--every", type=float, default=2.0, help="seconds between book summaries")
    parser.add_argument("--events", action="store_true", help="print every add/modify/delete")
    args = parser.parse_args()
    args.symbols = args.symbols or ["BTC/USD"]
    args.seconds = args.seconds or None
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        pass
