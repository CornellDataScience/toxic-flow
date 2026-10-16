"""Hand-built book scenarios for the fill simulators (OKR: unit-tested simulator)."""
import pytest

from sim.book import OrderBook
from sim.events import BookEvent as E, Trade as T
from sim.fill_sim import NaiveSimulator, QueueAwareSimulator


def setup(ahead):
    """Book with resting bids at 99 (ahead of us) and one ask at 101."""
    book = OrderBook()
    for oid, q in ahead:
        book.apply(E(0, "add", "bid", oid, 99, q))
    book.apply(E(0, "add", "ask", "ask1", 101, 10))
    qa, nv = QueueAwareSimulator(book), NaiveSimulator(book)
    qa.place("me", "bid", 99, 2, ts=1)
    nv.place("me", "bid", 99, 2, ts=1)
    return book, qa, nv


def feed(book, sims, *events):
    for ev in events:
        if isinstance(ev, E):
            prev = book.apply(ev)
            for s in sims:
                s.on_book_event(ev, prev)
        else:
            for s in sims:
                s.on_trade(ev)


def filled(sim):
    return sum(f.qty for f in sim.fills)


def test_queue_must_be_consumed_before_fill():
    book, qa, nv = setup([("A", 5)])
    feed(book, [qa, nv], T(2, 99, 3, "sell"), E(2, "modify", "bid", "A", 99, 2))
    assert filled(qa) == 0           # 2 still ahead of us
    assert filled(nv) == 2           # naive fills on first touch
    feed(book, [qa], T(3, 99, 3, "sell"), E(3, "delete", "bid", "A", 99, 0))
    assert filled(qa) == 1           # 2 consumed A, 1 left over reached us


def test_cancel_ahead_moves_us_up():
    book, qa, _ = setup([("A", 5)])
    feed(book, [qa], E(2, "delete", "bid", "A", 99, 0), T(3, 99, 1, "sell"))
    assert filled(qa) == 1


def test_orders_added_after_us_are_behind_us():
    book, qa, _ = setup([])
    feed(book, [qa], E(2, "add", "bid", "B", 99, 50), T(3, 99, 2, "sell"))
    assert filled(qa) == 2


def test_size_increase_ahead_moves_order_behind_us():
    book, qa, _ = setup([("A", 5)])
    feed(book, [qa], E(2, "modify", "bid", "A", 99, 8), T(3, 99, 1, "sell"))
    assert filled(qa) == 1


def test_trade_then_delete_is_not_double_counted():
    book, qa, _ = setup([("A", 5), ("B", 3)])
    feed(book, [qa], T(2, 99, 5, "sell"), E(2, "delete", "bid", "A", 99, 0))
    assert qa.orders["me"].qty_ahead == 3
    feed(book, [qa], T(3, 99, 2, "sell"), E(3, "modify", "bid", "B", 99, 1))
    assert filled(qa) == 0 and qa.orders["me"].qty_ahead == 1
    feed(book, [qa], T(4, 99, 2, "sell"), E(4, "delete", "bid", "B", 99, 0))
    assert filled(qa) == 1


def test_trade_through_our_price_fills_everything():
    book, qa, _ = setup([("A", 5)])
    feed(book, [qa], T(2, 98, 1, "sell"))
    assert filled(qa) == 2 and not qa.orders["me"].live


def test_buy_aggressor_does_not_fill_our_bid():
    book, qa, nv = setup([])
    feed(book, [qa, nv], T(2, 99, 5, "buy"))
    assert filled(qa) == filled(nv) == 0


def test_cannot_place_crossing_order():
    book, qa, _ = setup([])
    with pytest.raises(ValueError):
        qa.place("bad", "bid", 101, 1, ts=2)
