from decimal import Decimal

from toxicflow.kraken.book import L3Book, PairInfo
from toxicflow.kraken.eventlog import EVENT_COLUMNS, TRADE_COLUMNS, EventLogBuilder, TradeIndex, parse_ts_ns

SYM = "BTC/USD"
META = {"stream": "l3", "depth": 10, "pairs": {SYM: {"price_precision": 1, "qty_precision": 8}}}
S = 1_000_000_000
T0 = parse_ts_ns("2026-10-18T02:00:00Z")


def ts(sec: float) -> str:
    ns = T0 + round(sec * S)
    return f"2026-10-18T02:{(ns - T0) // S // 60:02d}:{(ns - T0) // S % 60:02d}.{(ns - T0) % S:09d}Z"


def o(oid, price, qty, event=None, at=None):
    d = {"order_id": oid, "limit_price": Decimal(price), "order_qty": Decimal(qty), "timestamp": at}
    return {**d, "event": event} if event else d


class Feed:
    """Recorded L3 records with real checksums, built by applying each item to a shadow L3Book."""

    def __init__(self):
        self.shadow = L3Book(PairInfo(SYM, 1, 8), 10)
        self.records = [(T0 - S, "meta", META)]

    def item(self, sec, msg_type, bids=(), asks=(), bad=False):
        item = {"symbol": SYM, "bids": list(bids), "asks": list(asks), "checksum": 0, "timestamp": ts(sec)}
        self.shadow.apply(msg_type, dict(item))
        self.shadow.synced = True
        item["checksum"] = self.shadow.checksum() ^ (1 if bad else 0)
        self.records.append((T0 + round(sec * S), "msg", {"channel": "level3", "type": msg_type, "data": [item]}))

    def snapshot(self, sec, bids, asks):
        self.shadow.reset()
        self.item(sec, "snapshot", bids, asks)


def public(trades, until_s=60):
    """Heartbeats every second (a healthy feed) plus trade messages: (sec, taker_side, price, qty, trade_id)."""
    recs = [(T0 + i * S, None) for i in range(-2, until_s)]
    for sec, side, price, qty, tid in trades:
        recs.append((T0 + round(sec * S), {"channel": "trade", "type": "update", "data": [
            {"symbol": SYM, "side": side, "price": Decimal(price), "qty": Decimal(qty), "ord_type": "market",
             "trade_id": tid, "timestamp": ts(sec)}]}))
    return sorted(recs, key=lambda r: r[0])


def build(feed, trade_records, **kw):
    rows = []
    b = EventLogBuilder(TradeIndex(trade_records), rows.append, **kw)
    for rec in feed.records:
        b.feed(*rec)
    b.finish()
    events = [dict(zip(EVENT_COLUMNS, r)) for r in rows]
    trades = [dict(zip(TRADE_COLUMNS, r)) for r in b.trade_rows()]
    return events, trades


def base_feed():
    f = Feed()
    f.snapshot(1, bids=[o("B1", "100.0", "1"), o("B2", "100.0", "2"), o("B3", "99.0", "5")],
               asks=[o("A1", "101.0", "3")])
    return f


def updates(events):
    return [(e["order_id"], e["event_type"], e["action"], e["evidence"]) for e in events if e["event_type"] != "snapshot"]


def test_fill_partial_fill_and_cancels():
    f = base_feed()
    f.item(2.010, "update", bids=[o("B1", "100.0", "1", "delete", ts(2.010))])     # traded
    f.item(2.020, "update", bids=[o("B2", "100.0", "1.5", "modify", ts(2.020))])   # 0.5 traded
    f.item(2.030, "update", bids=[o("B3", "99.0", "5", "delete", ts(2.030))])      # deep: cancel
    f.item(2.040, "update", asks=[o("A1", "101.0", "2", "modify", ts(2.040))])     # touch, no trade
    f.item(2.050, "update", bids=[o("B2", "100.0", "1.5", "delete", ts(2.050))])   # touch, no trade
    trades = public([(2.011, "sell", "100.0", "1", 7), (2.020, "sell", "100.0", "0.5", 8)])
    events, trade_rows = build(f, trades)

    assert updates(events) == [
        ("B1", "delete", "fill", "trade_match"),
        ("B2", "modify", "partial_fill", "trade_match"),
        ("B3", "delete", "cancel", "not_at_touch"),
        ("A1", "modify", "cancel", "no_trade"),
        ("B2", "delete", "cancel", "no_trade"),
    ]
    fill, partial = events[4], events[5]
    assert fill["trade_ids"] == "7" and fill["matched_qty"] == "1" and fill["match_dt_ns"] == "1000000"
    assert partial["trade_ids"] == "8" and partial["qty_delta"] == "-0.5" and partial["queue_pos"] == "0"
    assert all(e["valid"] == 1 for e in events)
    # trades point back at the events that consumed them
    assert [(t["trade_id"], t["maker_side"], t["matched_qty"], t["event_ids"], t["l3_valid"]) for t in trade_rows] == [
        (7, "bid", "1", str(fill["event_id"]), 1), (8, "bid", "0.5", str(partial["event_id"]), 1)]


def test_queue_state_on_snapshot_and_adds():
    f = base_feed()
    f.item(2, "update", bids=[o("B4", "100.0", "0.7", "add", ts(2))])
    f.item(3, "update", bids=[o("B1", "100.0", "4", "modify", ts(3))])  # qty up: back of the queue
    events, _ = build(f, public([]))
    snap = [(e["order_id"], e["queue_pos"], e["qty_ahead"], e["level_rank"]) for e in events[:4]]
    assert snap == [("B1", "0", "0", "0"), ("B2", "1", "1", "0"), ("B3", "0", "0", "1"), ("A1", "0", "0", "0")]
    add, up = events[4], events[5]
    assert (add["action"], add["queue_pos"], add["qty_ahead"], add["best_bid"]) == ("new", "2", "3", "100.0")
    assert (up["action"], up["evidence"], up["queue_pos"], up["qty_ahead"], up["prev_qty"]) == \
        ("refill", "qty_up", "2", "2.7", "1")


def test_readd_reentry_and_amend():
    f = base_feed()
    f.item(2, "update", asks=[o("A1", "101.0", "3", "delete", ts(2))])
    f.item(3, "update", asks=[o("A1", "101.0", "3", "add", ts(3))])             # same ID back: refill
    f.item(4, "update", asks=[o("A9", "105.0", "1", "add", ts(1.5))])           # old timestamp: reentry
    f.item(5, "update", bids=[o("B3", "98.5", "5", "modify", ts(5))])           # price moved
    events, _ = build(f, public([]))
    assert updates(events)[1:] == [
        ("A1", "add", "refill", "readd"),
        ("A9", "add", "reentry", "old_timestamp"),
        ("B3", "modify", "amend", "price_change"),
    ]


def test_no_trade_feed_means_unknown_not_cancel():
    f = base_feed()
    f.item(2, "update", bids=[o("B1", "100.0", "1", "delete", ts(2))])
    events, _ = build(f, [])  # no public recording at all
    assert updates(events) == [("B1", "delete", "unknown", "no_trade_coverage")]


def test_trades_cover_only_part_of_a_delete():
    f = base_feed()
    f.item(2, "update", bids=[o("B2", "100.0", "2", "delete", ts(2))])  # B2 is behind B1, but same price level
    f.item(3, "update", bids=[o("B1", "100.0", "1", "delete", ts(3))])
    events, trade_rows = build(f, public([(2, "sell", "100.0", "0.5", 1)]))
    assert updates(events) == [("B2", "delete", "unknown", "trades_partial"), ("B1", "delete", "cancel", "no_trade")]
    assert trade_rows[0]["matched_qty"] == "0"  # an unexplained trade is left unconsumed


def test_validity_unsynced_and_checksum_failure():
    f = Feed()
    f.item(0.5, "update", bids=[o("X", "100.0", "1", "add", ts(0.5))])  # before any snapshot
    f.records[-1][2]["data"][0]["checksum"] = 0
    f.shadow.reset()
    f.snapshot(1, bids=[o("B1", "100.0", "1")], asks=[o("A1", "101.0", "3")])
    f.item(2, "update", bids=[o("B1", "100.0", "1", "delete", ts(2))], bad=True)
    events, _ = build(f, public([(2, "sell", "100.0", "1", 1)]))
    assert [(e["order_id"], e["action"], e["evidence"], e["valid"]) for e in events] == [
        ("X", "unknown", "unsynced", 0),
        ("B1", "snapshot", "snapshot", 1), ("A1", "snapshot", "snapshot", 1),
        ("B1", "unknown", "book_invalid", 0),  # failed checksum: never consumes trades
    ]


def test_event_book_matches_l3book():
    """Replaying through EventL3Book gives the same checksums as L3Book (same queue logic)."""
    from toxicflow.kraken.eventlog import EventL3Book
    f = base_feed()
    f.item(2, "update", bids=[o("B4", "100.0", "0.7", "add"), o("B1", "100.0", "3", "modify"),
                              o("B2", "100.0", "1", "modify"), o("B3", "99.5", "5", "modify")],
           asks=[o("A2", "101.0", "1", "add"), o("A1", "101.0", "0", "delete")])
    a, b = L3Book(PairInfo(SYM, 1, 8), 10), EventL3Book(PairInfo(SYM, 1, 8), 10)
    for _, kind, msg in f.records:
        if kind == "msg":
            for item in msg["data"]:
                assert a.apply(msg["type"], item) is True and b.apply(msg["type"], item) is True
    assert a.orders == b.orders and len(b.events) == 6


def test_earlier_cancel_does_not_steal_a_later_fill_trade():
    f = base_feed()
    f.item(2.000, "update", bids=[o("B1", "100.0", "1", "delete", ts(2.000))])  # cancel at the touch
    f.item(2.020, "update", bids=[o("B2", "100.0", "2", "delete", ts(2.020))])  # filled 20 ms later
    events, trade_rows = build(f, public([(2.020, "sell", "100.0", "2", 1)]))
    assert updates(events) == [("B1", "delete", "cancel", "no_trade"), ("B2", "delete", "fill", "trade_match")]
    assert trade_rows[0]["event_ids"] == str(events[-1]["event_id"])
