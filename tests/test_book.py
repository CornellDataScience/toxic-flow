from sim.book import OrderBook
from sim.events import BookEvent as E


def test_add_modify_delete_and_mid():
    b = OrderBook()
    b.apply(E(0, "add", "bid", "a", 99, 2))
    b.apply(E(0, "add", "ask", "b", 101, 3))
    assert (b.best_bid(), b.best_ask(), b.mid(), b.spread()) == (99, 101, 100, 2)
    b.apply(E(1, "modify", "ask", "b", 101, 1))
    assert b.level_qty("ask", 101) == 1
    b.apply(E(2, "delete", "ask", "b", 101, 0))
    assert b.best_ask() is None and b.mid() is None


def test_fifo_order_and_priority_loss_on_increase():
    b = OrderBook()
    for oid in "xyz":
        b.apply(E(0, "add", "bid", oid, 99, 1))
    assert list(b.queue("bid", 99)) == ["x", "y", "z"]
    b.apply(E(1, "modify", "bid", "x", 99, 0.5))   # decrease keeps place
    assert list(b.queue("bid", 99)) == ["x", "y", "z"]
    b.apply(E(2, "modify", "bid", "x", 99, 5))     # increase -> back of queue
    assert list(b.queue("bid", 99)) == ["y", "z", "x"]


def test_unknown_order_counted_not_crash():
    b = OrderBook()
    b.apply(E(0, "delete", "bid", "ghost", 99, 0))
    assert b.unknown_events == 1
