from decimal import Decimal

import pytest
from execsim_helpers import BTC, Sealer, gz_member, level3, order, recording_lines, write_file

from toxicflow.execsim.reader import RecordingReader, book_items
from toxicflow.execsim.sim_book import ReadOnlyBookView, SimL3Book, SyncStatus

D = Decimal


def seeded(depth=10):
    """A synced SimL3Book and a Sealer holding the same state.

    bids 100.0: B1 1, B2 2 | 99.5: B3 5      asks 101.0: A1 3
    """
    sealer = Sealer(BTC, depth)
    book = SimL3Book(BTC, depth)
    snap = sealer.snapshot(
        bids=[order("B1", "100.0", "1"), order("B2", "100.0", "2"), order("B3", "99.5", "5")],
        asks=[order("A1", "101.0", "3")],
    )
    assert book.apply("snapshot", snap) is True
    return book, sealer


def only_event(book):
    assert len(book.last_events) == 1
    return book.last_events[0]


def test_snapshot_flags_and_no_events():
    book, _ = seeded()
    info = book.last_info
    assert info.snapshot_loaded and info.book_reset and not info.checksum_failed and not info.ignored_unsynced
    assert book.last_events == () and book.sync_status == SyncStatus.SYNCED


def test_add_goes_to_back_and_moves_best():
    book, sealer = seeded()
    item = sealer.update(bids=[order("B4", "100.0", "7", "add"), order("B5", "100.5", "1", "add")])
    assert book.apply("update", item) is True
    back, better = book.last_events
    assert (back.kind, back.qty_before, back.qty_after, back.queue_index_after) == ("add", None, D("7"), 2)
    assert back.requeued and back.best_bid_before == D("100.0") == back.best_bid_after
    assert better.best_bid_before == D("100.0") and better.best_bid_after == D("100.5")
    assert better.best_ask_before == better.best_ask_after == D("101.0")
    assert [e.event_index for e in book.last_events] == [0, 1]
    assert back.item_timestamp == item["timestamp"] and back.event_timestamp == item["bids"][0]["timestamp"]
    assert not book.last_info.snapshot_loaded and not book.last_info.book_reset


def test_modify_same_price_decrease_keeps_position():
    book, sealer = seeded()
    assert book.apply("update", sealer.update(bids=[order("B1", "100.0", "0.4", "modify")])) is True
    e = only_event(book)
    assert (e.qty_before, e.qty_after, e.reduction) == (D("1"), D("0.4"), D("0.6"))
    assert e.queue_index_before == e.queue_index_after == 0 and not e.requeued
    assert e.price_before == e.price_after == D("100.0")
    assert [oid for oid, _ in book.level_orders("bids", D("100.0"))] == ["B1", "B2"]


def test_modify_increase_keeps_position_like_l3book():
    book, sealer = seeded()
    book.apply("update", sealer.update(bids=[order("B1", "100.0", "9", "modify")]))
    e = only_event(book)
    assert e.queue_index_after == 0 and not e.requeued and e.reduction == 0


def test_price_changing_modify_requeues_at_new_level():
    book, sealer = seeded()
    assert book.apply("update", sealer.update(bids=[order("B1", "99.5", "1", "modify")])) is True
    e = only_event(book)
    assert (e.price_before, e.price_after) == (D("100.0"), D("99.5"))
    assert (e.queue_index_before, e.queue_index_after) == (0, 1) and e.requeued
    assert book.level_orders("bids", D("99.5")) == (("B3", D("5")), ("B1", D("1")))


def test_delete_removes_level_and_moves_best():
    book, sealer = seeded()
    item = sealer.update(asks=[order("A1", "101.0", "3", "delete"), order("A2", "102.0", "1", "add")])
    assert book.apply("update", item) is True
    gone, _ = book.last_events
    assert (gone.qty_before, gone.qty_after, gone.reduction) == (D("3"), None, D("3"))
    assert gone.best_ask_before == D("101.0") and gone.best_ask_after is None
    assert book.best_ask() == D("102.0") and D("101.0") not in book.asks


def test_delete_of_unknown_id_is_a_noop_event():
    book, sealer = seeded()
    assert book.apply("update", sealer.update(bids=[order("ZZ", "100.0", "1", "delete")])) is True
    e = only_event(book)
    assert e.noop and e.reduction == 0 and e.queue_index_before is None
    assert len(book.orders) == 4


def test_modify_of_unknown_id_is_appended():
    book, sealer = seeded()
    book.apply("update", sealer.update(bids=[order("NEW", "100.0", "1", "modify")]))
    e = only_event(book)
    assert e.qty_before is None and e.queue_index_after == 2 and e.requeued


def test_bids_events_apply_before_asks():
    book, sealer = seeded()
    item = sealer.update(asks=[order("A1", "101.0", "3", "delete")], bids=[order("B5", "100.5", "1", "add")])
    book.apply("update", item)
    assert [e.side for e in book.last_events] == ["bids", "asks"]
    assert book.last_events[1].best_bid_before == D("100.5")  # sees the bids event already applied


def test_depth_truncation_is_reported():
    book, sealer = seeded(depth=2)
    item = sealer.update(bids=[order("B9", "99.0", "1", "add"), order("B6", "100.2", "4", "add")])
    assert book.apply("update", item) is True
    deep, _ = book.last_events
    assert deep.dropped_by_truncation and deep.qty_after is None and not deep.requeued
    assert set(book.last_info.truncated) == {("bids", D("99.0"), 1), ("bids", D("99.5"), 1)}
    assert "B3" not in book.orders and "B9" not in book.orders and list(book.bids) == [D("100.2"), D("100.0")]


def test_unknown_event_kind_counted_not_applied():
    book, sealer = seeded()
    item = sealer.update(bids=[order("B1", "100.0", "1", "weird")])
    assert book.apply("update", item) is True
    assert book.last_events == () and book.event_counts["weird"] == 1


def test_checksum_failure_then_unsynced_until_snapshot():
    book, sealer = seeded()
    bad = sealer.update(bids=[order("B1", "100.0", "0.5", "modify")])
    bad["checksum"] += 1
    assert book.apply("update", bad) is False
    assert book.sync_status == SyncStatus.CHECKSUM_FAILED and book.last_info.checksum_failed
    assert len(book.last_events) == 1  # kept; caller must discard because result is not True

    assert book.apply("update", sealer.update(bids=[order("B2", "100.0", "1", "modify")])) is None
    assert book.last_info.ignored_unsynced and book.last_events == ()
    assert book.sync_status == SyncStatus.CHECKSUM_FAILED

    assert book.apply("snapshot", sealer.snapshot(bids=[order("B1", "100.0", "1")])) is True
    assert book.sync_status == SyncStatus.SYNCED and book.last_info.book_reset


def test_updates_before_first_snapshot_are_ignored():
    book = SimL3Book(BTC, 10)
    assert book.sync_status == SyncStatus.UNSYNCED
    assert book.apply("update", {"bids": [order("B1", "100.0", "1", "add")], "asks": [], "checksum": 0}) is None
    assert book.last_info.ignored_unsynced and not book.orders


def test_reader_to_book_drops_old_subscription_updates(tmp_path):
    # A rotated file starts with updates from the old subscription, then the snapshot (§4.6).
    old = Sealer()
    old.snapshot(bids=[order("X1", "90.0", "1")])
    stale = old.update(bids=[order("X2", "90.0", "1", "add")])
    sealer = Sealer()
    snap = sealer.snapshot(bids=[order("B1", "100.0", "1")], asks=[order("A1", "101.0", "1")])
    after = sealer.update(bids=[order("B2", "100.0", "2", "add")])
    msgs = [(1, level3("update", stale)), (2, {"channel": "heartbeat"}), (3, level3("snapshot", snap)),
            (4, level3("update", after))]
    write_file(tmp_path, "2026-10-18T03", gz_member(recording_lines(msgs)))

    book = SimL3Book(BTC, 10)
    results, positions = [], []
    for rec in RecordingReader(tmp_path):
        for msg_type, item in book_items(rec):
            results.append(book.apply(msg_type, item, position=rec))
            positions.extend((e.position.seq, e.position.line_no) for e in book.last_events)
    assert results == [None, True, True]
    assert positions == [(4, 5)]
    assert book.level_orders("bids", D("100.0")) == (("B1", D("1")), ("B2", D("2")))


def test_read_only_view():
    book, sealer = seeded()
    view = book.view()
    assert isinstance(view, ReadOnlyBookView) and view.synced and view.symbol == "BTC/USD"
    assert (view.best_bid(), view.best_ask(), view.mid()) == (D("100.0"), D("101.0"), D("100.5"))
    assert view.level_qty("bids", D("100.0")) == D("3") and view.level_qty("bids", D("1")) == 0
    assert view.levels("bids") == ((D("100.0"), D("3"), 2), (D("99.5"), D("5"), 1))
    assert view.levels("asks", 1) == ((D("101.0"), D("3"), 1),)
    orders = view.level_orders("bids", D("100.0"))
    assert orders == (("B1", D("1")), ("B2", D("2")))
    with pytest.raises(AttributeError):
        view.bids  # noqa: B018 - no access to the underlying containers
    with pytest.raises(AttributeError):
        view.extra = 1
    book.apply("update", sealer.update(bids=[order("B1", "100.0", "1", "delete")]))
    assert orders == (("B1", D("1")), ("B2", D("2")))  # earlier copy unchanged
    assert view.level_orders("bids", D("100.0")) == (("B2", D("2")),)  # live view follows the book
