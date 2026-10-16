"""Replay an event stream through the book, simulators and a quoting strategy."""
import pandas as pd

from .book import OrderBook
from .events import BookEvent
from .fill_sim import NaiveSimulator, QueueAwareSimulator


class JoinBestQuoter:
    """Simplest maker: rest `size` at the best bid and best ask.

    Re-quotes when its order fills or when the best price moves away.
    Each simulator gets its own copy of the order, so naive vs queue-aware
    are compared on the same quoting logic.
    """

    def __init__(self, size=1.0, min_interval=0.5):
        self.size = size
        self.min_interval = min_interval
        self.n = 0
        self.last_quote = {}  # (sim name, side) -> ts

    def on_event(self, ts, book, sims):
        best = {"bid": book.best_bid(), "ask": book.best_ask()}
        if best["bid"] is None or best["ask"] is None:
            return
        for sim in sims:
            for side in ("bid", "ask"):
                mine = [o for o in sim.live_orders() if o.side == side]
                if mine and mine[0].price == best[side]:
                    continue
                if ts - self.last_quote.get((sim.name, side), -1e18) < self.min_interval:
                    continue
                for o in mine:
                    sim.cancel(o.oid)
                self.n += 1
                sim.place(f"{sim.name}-{self.n}", side, best[side], self.size, ts)
                self.last_quote[(sim.name, side)] = ts


def run_backtest(events, strategy=None):
    """Returns (fills_by_sim: dict[name -> DataFrame], mids: DataFrame)."""
    book = OrderBook()
    sims = [QueueAwareSimulator(book), NaiveSimulator(book)]
    mid_ts, mid_val, last_mid = [], [], None

    for ev in events:
        if isinstance(ev, BookEvent):
            prev = book.apply(ev)
            for s in sims:
                s.on_book_event(ev, prev)
            m = book.mid()
            if m is not None and m != last_mid:
                mid_ts.append(ev.ts); mid_val.append(m); last_mid = m
        else:
            for s in sims:
                s.on_trade(ev)
        if strategy is not None:
            strategy.on_event(ev.ts, book, sims)

    fills = {s.name: pd.DataFrame([f.__dict__ for f in s.fills]) for s in sims}
    mids = pd.DataFrame({"ts": mid_ts, "mid": mid_val})
    return fills, mids
