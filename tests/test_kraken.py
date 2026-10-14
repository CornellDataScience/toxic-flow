from decimal import Decimal

from toxicflow.kraken.book import L2Book, L3Book, PairInfo
from toxicflow.kraken.checksum import crc32
from toxicflow.kraken.rest import sign
from toxicflow.kraken.ws import loads

BTC = PairInfo("BTC/USD", price_precision=1, qty_precision=8)

# Worked example from https://docs.kraken.com/api/docs/guides/spot-ws-book-v2
DOC_BIDS = [
    ("45283.5", "0.10000000"), ("45283.4", "1.54582015"), ("45282.1", "0.10000000"),
    ("45281.0", "0.10000000"), ("45280.3", "1.54592586"), ("45279.0", "0.07990000"),
    ("45277.6", "0.03310103"), ("45277.5", "0.30000000"), ("45277.3", "1.54602737"),
    ("45276.6", "0.15445238"),
]
DOC_ASKS = [
    ("45285.2", "0.00100000"), ("45286.4", "1.54571953"), ("45286.6", "1.54571109"),
    ("45289.6", "1.54560911"), ("45290.2", "0.15890660"), ("45291.8", "1.54553491"),
    ("45294.7", "0.04454749"), ("45296.1", "0.35380000"), ("45297.5", "0.09945542"),
    ("45299.5", "0.18772827"),
]


def levels(pairs):
    # Real messages carry JSON numbers, so trailing zeros are lost; mimic that.
    return [{"price": Decimal(p).normalize(), "qty": Decimal(q).normalize()} for p, q in pairs]


def test_l2_checksum_matches_kraken_doc_example():
    book = L2Book(BTC, depth=10)
    assert book.apply("snapshot", {"bids": levels(DOC_BIDS), "asks": levels(DOC_ASKS), "checksum": 3310070434})


def test_l2_update_remove_and_truncate():
    book = L2Book(BTC, depth=10)
    book.apply("snapshot", {"bids": levels(DOC_BIDS), "asks": levels(DOC_ASKS), "checksum": 3310070434})
    # remove best bid, add a new best ask -> the 11th ask must be truncated
    update = {"bids": [{"price": Decimal("45283.5"), "qty": Decimal(0)}],
              "asks": [{"price": Decimal("45284.0"), "qty": Decimal("2")}], "checksum": 0}
    book.apply("update", update)
    assert book.best_bid() == Decimal("45283.4")
    assert book.best_ask() == Decimal("45284.0")
    assert len(book.asks) == 10 and Decimal("45299.5") not in book.asks
    assert not book.synced  # checksum 0 was wrong -> must resync


def test_updates_ignored_before_snapshot():
    book = L2Book(BTC, depth=10)
    assert book.apply("update", {"bids": levels(DOC_BIDS), "asks": [], "checksum": 0}) is None
    assert not book.bids


def test_json_numbers_parse_as_decimal():
    msg = loads('{"price": 45283.5, "qty": 0.1}')
    assert msg["price"] == Decimal("45283.5") and isinstance(msg["qty"], Decimal)


def l3_order(oid, price, qty, event=None):
    o = {"order_id": oid, "limit_price": Decimal(price), "order_qty": Decimal(qty), "timestamp": "t"}
    return {**o, "event": event} if event else o


def l3_checksum_str(asks, bids):
    """Hand-built checksum input: one price+qty per order, asks then bids."""
    def f(p, q):
        return f"{Decimal(p):.1f}".replace(".", "").lstrip("0") + f"{Decimal(q):.8f}".replace(".", "").lstrip("0")
    return "".join(f(p, q) for p, q in asks) + "".join(f(p, q) for p, q in bids)


def test_l3_events_and_checksum():
    book = L3Book(BTC, depth=10)
    snap = {
        "bids": [l3_order("B1", "100.0", "1"), l3_order("B2", "100.0", "2"), l3_order("B3", "99.5", "5")],
        "asks": [l3_order("A1", "101.0", "3")],
    }
    snap["checksum"] = crc32(l3_checksum_str([("101.0", "3")], [("100.0", "1"), ("100.0", "2"), ("99.5", "5")]))
    assert book.apply("snapshot", snap)

    # B1 partially filled (keeps its position), B4 added at 100.0, B3 cancelled
    update = {
        "bids": [l3_order("B1", "100.0", "0.4", "modify"), l3_order("B4", "100.0", "7", "add"),
                 l3_order("B3", "99.5", "5", "delete")],
        "asks": [],
    }
    update["checksum"] = crc32(l3_checksum_str([("101.0", "3")], [("100.0", "0.4"), ("100.0", "2"), ("100.0", "7")]))
    assert book.apply("update", update)
    assert list(book.bids[Decimal("100.0")]) == ["B1", "B2", "B4"]
    assert Decimal("99.5") not in book.bids and "B3" not in book.orders
    assert book.event_counts == {"modify": 1, "add": 1, "delete": 1}


def test_l3_depth_truncation_forgets_orders():
    book = L3Book(BTC, depth=1)
    snap = {"bids": [l3_order("B1", "100.0", "1")], "asks": [l3_order("A1", "101.0", "1")], "checksum": 0}
    book.apply("snapshot", snap)
    book.synced = True  # snapshot checksum is irrelevant here
    book.apply("update", {"bids": [l3_order("B2", "100.5", "1", "add")], "asks": [], "checksum": 0})
    assert list(book.bids) == [Decimal("100.5")] and "B1" not in book.orders


def test_sign_matches_kraken_doc_example():
    # https://docs.kraken.com/api/docs/guides/spot-rest-auth
    secret = "kQH5HW/8p1uGOVjbgWA7FunAmGO8lsSUXNsu3eow76sz84Q18fWxnyRzBHCd3pd5nE9qa99HAZtuZuj6F1huXg=="
    data = {"nonce": "1616492376594", "ordertype": "limit", "pair": "XBTUSD", "price": 37500, "type": "buy", "volume": 1.25}
    assert sign("/0/private/AddOrder", data, secret) == (
        "4/dpxb3iT4tp/ZCVEwSnEsLxx0bqyhLpdfOpc6fn7OR8+UClSV5n9E6aSS8MPtnRfp32bAb0nmbRn6H8ndwLUQ=="
    )
