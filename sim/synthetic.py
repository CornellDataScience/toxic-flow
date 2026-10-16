"""Synthetic L3 event stream, so we can build and test before real data arrives.

A hidden "fair value" random-walks. Noise traders send random market orders;
informed traders buy when fair value > mid and sell when below. Informed flow
consumes the book and moves the price, so passive fills get adversely
selected, which is exactly what our pipeline should detect.

Prices are in integer ticks (start 10000) to avoid float-equality issues.
Real data will use exchange prices; keep that in mind when comparing levels.
"""
import numpy as np

from .book import OrderBook
from .events import BookEvent, Trade


def generate(n_events=50_000, seed=0, p_add=0.55, p_cancel=0.30,
             informed_share=0.4, fv_vol=0.3, start=10_000.0):
    rng = np.random.default_rng(seed)
    book, events = OrderBook(), []
    t, fv, next_id = 0.0, start, 0

    def emit(ev):
        events.append(ev)
        if isinstance(ev, BookEvent):
            book.apply(ev)

    def add(side, price, qty):
        nonlocal next_id
        next_id += 1
        emit(BookEvent(t, "add", side, f"o{next_id}", float(price), float(qty)))

    # seed a book: 10 levels each side
    for k in range(1, 11):
        for _ in range(3):
            add("bid", start - k, rng.integers(1, 6))
            add("ask", start + k, rng.integers(1, 6))

    while len(events) < n_events:
        t += rng.exponential(0.05)
        fv += rng.normal(0, fv_vol)
        bb, ba = book.best_bid(), book.best_ask()
        if bb is None:
            add("bid", (ba or fv) - 1, 3); continue
        if ba is None:
            add("ask", bb + 1, 3); continue

        r = rng.random()
        if r < p_add:
            side = "bid" if rng.random() < 0.5 else "ask"
            offset = rng.geometric(0.5) - 1          # 0 = join best
            if side == "bid":
                price = bb - offset
                if offset == 0 and ba - bb > 1 and rng.random() < 0.3:
                    price = bb + 1                   # improve inside the spread
            else:
                price = ba + offset
                if offset == 0 and ba - bb > 1 and rng.random() < 0.3:
                    price = ba - 1
            add(side, price, rng.integers(1, 6))

        elif r < p_add + p_cancel:
            side = "bid" if rng.random() < 0.5 else "ask"
            prices = list(book.levels[side])
            price = prices[rng.integers(len(prices))]
            ids = list(book.levels[side][price])
            oid = ids[rng.integers(len(ids))]
            emit(BookEvent(t, "delete", side, oid, price, 0.0))

        else:  # market order
            mid = book.mid()
            if rng.random() < informed_share:
                buy = fv > mid
            else:
                buy = rng.random() < 0.5
            vol, opp = float(rng.integers(1, 10)), ("ask" if buy else "bid")
            while vol > 0:
                best = book.best_ask() if buy else book.best_bid()
                if best is None:
                    break
                oid, q = next(iter(book.levels[opp][best].items()))
                take = min(q, vol)
                # Trade first, then the book update (the ordering our simulator assumes)
                emit(Trade(t, best, take, "buy" if buy else "sell"))
                if take >= q:
                    emit(BookEvent(t, "delete", opp, oid, best, 0.0))
                else:
                    emit(BookEvent(t, "modify", opp, oid, best, q - take))
                vol -= take
    return events
