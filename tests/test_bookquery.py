import random
import warnings
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from toxicflow.bookquery import BookQuery, InvalidTimeError, OutsideDepthWarning, build_tables
from toxicflow.bookquery.units import decimal_to_ticks, parse_iso_ns
from toxicflow.kraken.book import L3Book, PairInfo
from toxicflow.kraken.recorder import read_recording

PAIR = PairInfo("TEST/USD", price_precision=1, qty_precision=0)
D = Decimal


def make_events(rows):
    """rows: (ts, order_id, side, price, qty, event); seq = row order, one message per row."""
    df = pd.DataFrame(rows, columns=["ts", "order_id", "side", "price", "qty", "event"])
    df["seq"] = np.arange(len(df))
    return df


def book_state(book):
    """Comparable full state: per side, levels best first, each an ordered list of (oid, lots)."""
    return [[(p, list(lvl.items())) for p, lvl in side.items()] for side in (book.bids, book.asks)]


# A known sequence: B sits behind A at 100.0, then A is partly filled,
# cancelled, D joins, B is enlarged (loses its place), then the level empties.
SEQ = [
    (10, "A", "bid", "100.0", 5, "snapshot"),
    (10, "B", "bid", "100.0", 3, "snapshot"),
    (10, "C", "bid", "99.5", 2, "snapshot"),
    (10, "X", "ask", "101.0", 4, "snapshot"),
    (20, "A", "bid", "100.0", 2, "modify"),   # partial fill: A keeps the front
    (30, "A", "bid", "100.0", 2, "delete"),   # cancel ahead of B
    (35, "D", "bid", "100.0", 1, "add"),
    (40, "B", "bid", "100.0", 6, "modify"),   # size up: B goes behind D
    (50, "B", "bid", "100.0", 6, "delete"),
    (50, "D", "bid", "100.0", 1, "delete"),   # level gone -> best bid 99.5
]


@pytest.fixture
def bq():
    return BookQuery.load(make_events(SEQ), pair=PAIR)


def test_partial_fill_keeps_queue_position(bq):
    assert bq.queue_at("bid", "100.0", 10) == [("A", D(5)), ("B", D(3))]
    assert bq.queue_at("bid", "100.0", 20) == [("A", D(2)), ("B", D(3))]
    assert bq.queue_ahead("bid", "100.0", 10, "B") == D(5)
    assert bq.queue_ahead("bid", "100.0", 20, "B") == D(2)  # fill shrank the size ahead
    assert bq.queue_ahead("bid", "100.0", 20, "A") == D(0)


def test_cancel_ahead_shrinks_size_ahead(bq):
    assert bq.queue_ahead("bid", "100.0", 30, "B") == D(0)
    assert bq.queue_ahead("bid", "100.0", 30) == D(3)  # a new order would join behind B


def test_size_increase_moves_to_back(bq):
    assert bq.queue_at("bid", "100.0", 35) == [("B", D(3)), ("D", D(1))]
    assert bq.queue_at("bid", "100.0", 40) == [("D", D(1)), ("B", D(6))]
    assert bq.queue_ahead("bid", "100.0", 40, "B") == D(1)


def test_full_removal_moves_best(bq):
    assert bq.best_at(40).bid_price == D("100.0")
    best = bq.best_at(50)
    assert best.bid_price == D("99.5") and best.bid_qty == D(2)
    assert bq.mid_at(50) == pytest.approx(100.25)
    assert bq.spread_at(50) == pytest.approx(1.5)
    assert bq.queue_at("bid", "100.0", 50) == []


def test_queue_ahead_unknown_order_raises(bq):
    with pytest.raises(KeyError):
        bq.queue_ahead("bid", "100.0", 30, "A")


def test_equal_timestamps_apply_in_seq_order():
    rows = SEQ[:4] + [(20, "E", "bid", "100.5", 1, "add"), (20, "E", "bid", "100.5", 1, "delete")]
    bq = BookQuery.load(make_events(rows), pair=PAIR)
    assert bq.best_at(20).bid_price == D("100.0")  # add then delete -> gone
    rows = SEQ[:4] + [(20, "E", "bid", "100.5", 1, "delete"), (20, "E", "bid", "100.5", 1, "add")]
    bq = BookQuery.load(make_events(rows), pair=PAIR)
    assert bq.best_at(20).bid_price == D("100.5")  # delete (no-op) then add -> present


def test_seq_not_row_order_decides():
    df = make_events(SEQ).iloc[::-1]  # rows shuffled; seq still says the real order
    bq = BookQuery.load(df, pair=PAIR)
    assert bq.queue_at("bid", "100.0", 40) == [("D", D(1)), ("B", D(6))]


@pytest.mark.parametrize("t,expected_front_qty", [(19, 5), (20, 2), (21, 2)])
def test_as_of_boundary(bq, t, expected_front_qty):
    # the modify at ts=20 is visible exactly at 20, not 1 ns before
    assert bq.queue_at("bid", "100.0", t)[0] == ("A", D(expected_front_qty))
    assert bq.best_at(t).bid_qty == D(expected_front_qty + 3)
    assert bq.book_at(t).ts == (10 if t < 20 else 20)


def test_vectorized_matches_scalar(bq):
    ts = np.array([9, 10, 19, 20, 21, 35, 50, 51])
    mids = bq.mid_at(ts)
    for t, m in zip(ts, mids):
        s = bq.mid_at(int(t))
        assert (np.isnan(m) and np.isnan(s)) or m == s
    df = bq.best_at(ts)
    assert list(df["bid_qty"].iloc[2:5]) == [8.0, 5.0, 5.0]


# -------------------------------------------------------------- validity

TWO_INTERVALS = SEQ + [
    (70, "Z", "ask", "90.0", 1, "add"),       # garbage between intervals: must be ignored
    (100, "P", "bid", "200.0", 1, "snapshot"),
    (100, "Q", "ask", "201.0", 1, "snapshot"),
    (150, "P", "bid", "200.0", 1, "delete"),
]
INTERVALS = pd.DataFrame({"start": [10, 100], "end": [55, 200]})


@pytest.fixture
def bq2():
    return BookQuery.load(make_events(TWO_INTERVALS), valid_intervals=INTERVALS, pair=PAIR)


def test_outside_intervals_is_none_or_nan(bq2):
    for t in (9, 56, 70, 99, 201):
        assert not bq2.is_valid(t)
        assert bq2.book_at(t) is None and bq2.best_at(t) is None and bq2.queue_at("bid", "100.0", t) is None
        assert np.isnan(bq2.mid_at(t)) and np.isnan(bq2.spread_at(t))
    assert list(bq2.is_valid([9, 10, 55, 56, 100, 200, 201])) == [False, True, True, False, True, True, False]
    assert np.isnan(bq2.best_at(np.array([56, 70])).to_numpy()).all()


def test_valid_end_is_inclusive_and_stays_as_of(bq2):
    # no events between 50 and 55, still valid and shows the ts=50 state
    assert bq2.best_at(55).bid_price == D("99.5")


def test_strict_raises(bq2):
    strict = BookQuery.load(make_events(TWO_INTERVALS), valid_intervals=INTERVALS, pair=PAIR, strict=True)
    with pytest.raises(InvalidTimeError):
        strict.mid_at(70)
    with pytest.raises(InvalidTimeError):
        strict.book_at(70)
    with pytest.raises(InvalidTimeError):
        strict.mid_at(np.array([50, 70]))
    assert strict.is_valid(70) is False  # is_valid never raises


def test_book_rebuilt_fresh_per_interval(bq2):
    book = bq2.book_at(100)
    assert set(book.orders) == {"P", "Q"}  # nothing from the first interval, nor the ts=70 event
    assert bq2.best_at(150).bid_price is None and np.isnan(bq2.mid_at(150))  # empty bid side


def test_mid_after_does_not_bridge_gaps(bq2):
    out = bq2.mid_after(np.array([20, 50, 120, 60]), horizons=[5, 10, 80])
    assert out.loc[50, "5"] == pytest.approx(100.25)       # 55 is still inside
    assert np.isnan(out.loc[50, "10"])                      # 60 is in the gap
    assert np.isnan(out.loc[50, "80"])                      # 130 is valid, but another interval
    assert out.loc[20, "10"] == pytest.approx(bq2.mid_at(30))
    assert np.isnan(out.loc[120, "80"])                     # 200 is valid, but the bid side is empty by then
    assert out.loc[120, "10"] == pytest.approx(200.5)
    assert out.loc[60].isna().all()                         # t itself invalid
    scalar = bq2.mid_after(20, horizons=("10ns",))
    assert scalar["10ns"] == pytest.approx(bq2.mid_at(30))


def test_iter_book_skips_gaps_and_resets(bq2):
    steps = list(bq2.iter_book(0, 1000))
    assert [s.ts for s in steps] == [10, 20, 30, 35, 40, 50, 100, 150]
    assert [s.reset for s in steps] == [True] + [False] * 5 + [True, False]
    assert [s.interval for s in steps] == [0] * 6 + [1] * 2


def test_iter_book_starts_as_of_t0(bq2):
    steps = list(bq2.iter_book(25, 45))
    assert steps[0].ts == 20 and steps[0].reset
    assert [s.ts for s in steps[1:]] == [30, 35, 40]
    assert book_state(steps[-1].book) == book_state(bq2.book_at(45))


# ----------------------------------------------------------------- depth

def test_depth_truncation_and_warning():
    rows = [
        (10, "B1", "bid", "100.0", 1, "snapshot"),
        (10, "B2", "bid", "99.0", 1, "snapshot"),
        (10, "A1", "ask", "101.0", 1, "snapshot"),
        (20, "B3", "bid", "100.5", 1, "add"),  # pushes 99.0 past depth=2
    ]
    bq = BookQuery.load(make_events(rows), pair=PAIR, depth=2)
    assert "B2" not in bq.book_at(20).orders
    with pytest.warns(OutsideDepthWarning):
        assert bq.queue_at("bid", "99.0", 20) == []
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bq.queue_at("bid", "100.0", 20)   # inside the visible depth
        bq.queue_at("ask", "105.0", 20)   # ask side has < depth levels: fully visible


# ----------------------------------------------- checkpoints vs full replay

def random_events(n, seed):
    rng = random.Random(seed)
    rows, live, ts = [], {}, 0
    for k in range(n):
        ts += rng.choice([0, 0, 1, 3])  # plenty of equal timestamps
        r = rng.random()
        if not live or r < 0.45:
            side = rng.choice(["bid", "ask"])
            price = rng.randint(990, 1000) if side == "bid" else rng.randint(1001, 1011)
            oid = f"o{k}"
            live[oid] = (side, price)
            rows.append((ts, oid, side, D(price).scaleb(-1), rng.randint(1, 9), "add"))
        else:
            oid = rng.choice(sorted(live))
            side, price = live[oid]
            if r < 0.7:
                rows.append((ts, oid, side, D(price).scaleb(-1), rng.randint(1, 9), "modify"))
            else:
                del live[oid]
                rows.append((ts, oid, side, D(price).scaleb(-1), 0, "delete"))
    return make_events(rows)


@pytest.mark.parametrize("depth", [None, 5])
def test_checkpoints_equal_full_replay(depth):
    ev = random_events(3000, seed=1)
    fast = BookQuery.load(ev, pair=PAIR, depth=depth, checkpoint_every=37)
    slow = BookQuery.load(ev, pair=PAIR, depth=depth, checkpoint_every=10**9)
    assert sum(len(c) for c in fast._ckpts) > 10 and not any(slow._ckpts)
    rng = random.Random(2)
    t_max = int(ev["ts"].max())
    for t in [rng.randint(0, t_max) for _ in range(200)] + [0, t_max]:
        a, b = fast.book_at(t), slow.book_at(t)
        assert book_state(a) == book_state(b) and a.ts == b.ts
        assert fast.best_at(t) == a.best()


def test_cache_roundtrip(tmp_path):
    ev = random_events(500, seed=3)
    first = BookQuery.load(ev, pair=PAIR, checkpoint_every=50, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("bookquery_*.pkl"))) == 1
    second = BookQuery.load(ev, pair=PAIR, checkpoint_every=50, cache_dir=tmp_path)
    for k in first._tob:
        np.testing.assert_array_equal(first._tob[k], second._tob[k])
    assert book_state(first.book_at(200)) == book_state(second.book_at(200))
    BookQuery.load(ev.iloc[:-1], pair=PAIR, checkpoint_every=50, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("bookquery_*.pkl"))) == 2  # different input -> different key


def test_load_rejects_bad_input():
    bad = make_events(SEQ)
    bad.loc[5, "ts"] = 5  # goes backwards in seq order
    with pytest.raises(ValueError):
        BookQuery.load(bad, pair=PAIR)
    with pytest.raises(ValueError):
        BookQuery.load(make_events(SEQ), valid_intervals=pd.DataFrame({"start": [10, 30], "end": [40, 50]}), pair=PAIR)


def test_units():
    assert parse_iso_ns("2026-10-16T21:07:35.159647442Z") == pd.Timestamp("2026-10-16T21:07:35.159647442Z").value
    assert parse_iso_ns("2026-10-16T21:07:38.790402Z") == pd.Timestamp("2026-10-16T21:07:38.790402Z").value
    assert decimal_to_ticks(81537.4, 1) == 815374
    assert decimal_to_ticks(D("0.00045992"), 8) == 45992
    with pytest.raises(ValueError):
        decimal_to_ticks(D("1.05"), 1)


# --------------------------------------------------------- real recordings

L3_FILES = sorted(Path(__file__).resolve().parents[1].glob("data/raw/l3/*/*.jsonl.gz"))
PUBLIC_FILES = sorted(Path(__file__).resolve().parents[1].glob("data/raw/public/*/*.jsonl.gz"))
needs_data = pytest.mark.skipif(not L3_FILES, reason="no L3 recordings under data/raw/l3")
SYMBOL = "BTC/USD"


@pytest.fixture(scope="module")
def real():
    tables = build_tables(L3_FILES, PUBLIC_FILES, symbol=SYMBOL)
    bq = BookQuery.load(tables.events, tables.trades, tables.valid_intervals,
                        pair=tables.pair, depth=tables.depth, checkpoint_every=500)
    return tables, bq


def raw_messages(sample=frozenset()):
    """(exchange ts, exchange checksum, levels) per L3 message, in feed order.

    levels: for message indices in `sample`, the existing L3Book's aggregated
    (price, size) per level after that message, or None if it was unsynced.
    """
    meta, book, k = None, None, 0
    for path in L3_FILES:
        for _, kind, payload in read_recording(path):
            if kind == "meta":
                meta, book = payload, None
                continue
            if payload.get("channel") != "level3" or payload.get("type") not in ("snapshot", "update"):
                continue
            for item in payload["data"]:
                if item["symbol"] != SYMBOL:
                    continue
                if book is None:
                    p = meta["pairs"][SYMBOL]
                    book = L3Book(PairInfo(SYMBOL, p["price_precision"], p["qty_precision"]), meta["depth"])
                book.apply(payload["type"], item)
                levels = None
                if k in sample and book.synced:
                    levels = [[(p, sum(lvl.values())) for p, lvl in book.side(n).items()] for n in ("bids", "asks")]
                yield parse_iso_ns(item["timestamp"]), item["checksum"], levels
                k += 1


@needs_data
def test_real_every_checksum_matches(real):
    tables, _ = real
    assert tables.stats["checksum_fail"] == 0 and tables.stats["checksum_ok"] > 0


@needs_data
def test_real_book_matches_exchange_checksum_and_existing_replay(real):
    """At sampled message times, book_at must match Kraken's checksum exactly, and
    level sizes must match the existing L3Book (which differs only in queue order
    after a size increase)."""
    _, bq = real
    last_at_ts = {}
    for k, (ts, _, _) in enumerate(raw_messages()):
        last_at_ts[ts] = k  # the as-of state at ts is after the last message with that ts
    sampled = set(sorted(last_at_ts.values())[::40])
    compared = 0
    for k, (ts, checksum, ref_levels) in enumerate(raw_messages(frozenset(sampled))):
        if k not in sampled or not bq.is_valid(ts):
            continue
        ours = bq.book_at(ts)
        assert ours.checksum() == checksum
        if ref_levels is not None:
            our_levels = [[(ours.px(p), ours.qty(sum(lvl.values()))) for p, lvl in side.items()]
                          for side in (ours.bids, ours.asks)]
            assert our_levels == ref_levels
            compared += 1
    assert compared > 100


@needs_data
def test_real_top_of_book_series_matches_book_at(real):
    _, bq = real
    rng = np.random.default_rng(0)
    lo, hi = bq.valid_intervals[0][0], bq.valid_intervals[-1][1]
    for t in rng.integers(lo, hi, 300):
        book = bq.book_at(int(t))
        if book is None:
            assert np.isnan(bq.mid_at(int(t)))
            continue
        assert bq.best_at(int(t)) == book.best()
        assert bq.mid_at(int(t)) == pytest.approx(float(book.best().mid), abs=1e-9)


@needs_data
def test_real_checkpoints_equal_full_replay(real):
    tables, bq = real
    slow = BookQuery.load(tables.events, valid_intervals=tables.valid_intervals,
                          pair=tables.pair, depth=tables.depth, checkpoint_every=10**9)
    rng = np.random.default_rng(1)
    lo, hi = bq.valid_intervals[0][0], bq.valid_intervals[-1][1]
    for t in rng.integers(lo, hi, 100):
        a, b = bq.book_at(int(t)), slow.book_at(int(t))
        assert (a is None and b is None) or book_state(a) == book_state(b)


@needs_data
def test_real_iter_book_ends_at_book_at(real):
    _, bq = real
    t0 = bq.valid_intervals[0][0]
    t1 = t0 + 20 * 10**9
    last = None
    for step in bq.iter_book(t0, t1):
        last = step
    assert book_state(last.book) == book_state(bq.book_at(t1))


@needs_data
def test_real_recv_clock(real):
    tables, _ = real
    bq = BookQuery.load(tables.events, tables.trades, tables.valid_intervals,
                        pair=tables.pair, depth=tables.depth, clock="recv")
    t = int(tables.events["ts_recv"].iloc[5000])
    assert bq.is_valid(t)
    assert len(bq.trades_between(t, t + 10**10)) == ((tables.trades["ts_recv"] > t) &
                                                      (tables.trades["ts_recv"] <= t + 10**10)).sum()
