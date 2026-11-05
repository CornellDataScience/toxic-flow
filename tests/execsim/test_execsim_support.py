"""timeutil, SimConfig, SimL3Book accessors, BookView, JoinBestStrategy and PlacementSchedule."""

from decimal import Decimal

import pytest
from execsim_helpers import BTC, Sealer, order

from toxicflow.execsim.config import SimConfig
from toxicflow.execsim.models import BookView, LevelSummary, OrderIntent, PlacementRequest, Side, Skip, SkipReason
from toxicflow.execsim.schedule import PlacementSchedule
from toxicflow.execsim.sim_book import SimL3Book
from toxicflow.execsim.strategies import JoinBestStrategy
from toxicflow.execsim.timeutil import S, default_tick, format_ns, parse_rfc3339_ns

D = Decimal


# --- timeutil -------------------------------------------------------------


def test_parse_is_nanosecond_exact():
    base = parse_rfc3339_ns("2026-10-18T02:06:55Z")
    assert base == 1792289215 * S
    assert parse_rfc3339_ns("2026-10-18T02:06:55.103101894Z") == base + 103_101_894
    assert parse_rfc3339_ns("2026-10-18T02:06:55.1Z") == base + 100_000_000
    assert parse_rfc3339_ns("2026-10-18T04:06:55.5+02:00") == base + 500_000_000
    assert parse_rfc3339_ns("2026-10-18T02:06:55.000000001Z") - base == 1


@pytest.mark.parametrize("bad", [None, "", "2026-10-18 02:06:55Z", "2026-10-18T02:06:55.1234567890Z",
                                 "2026-10-18T02:06:55", "2026-02-30T00:00:00Z", 1792289215])
def test_parse_rejects_malformed(bad):
    with pytest.raises(ValueError):
        parse_rfc3339_ns(bad)


def test_format_round_trips():
    ts = "2026-10-18T02:06:55.103101894Z"
    assert format_ns(parse_rfc3339_ns(ts)) == ts
    assert format_ns(parse_rfc3339_ns("2026-10-18T02:06:55Z")) == "2026-10-18T02:06:55.000000000Z"


def test_default_tick():
    assert default_tick(1) == D("0.1") and default_tick(2) == D("0.01") and default_tick(0) == D("1")


# --- SimConfig --------------------------------------------------------------


def test_config_coerces_and_defaults():
    cfg = SimConfig(symbols=["BTC/USD"], order_qty={"BTC/USD": 0.001}, horizon_ns=10 * S, tick_size={"BTC/USD": "0.1"})
    assert cfg.symbols == ("BTC/USD",) and cfg.order_qty["BTC/USD"] == D("0.001")
    assert cfg.tick_size == {"BTC/USD": D("0.1")}
    assert (cfg.latency_ns, cfg.gap_ns, cfg.increase_policy, cfg.view_depth) == (0, 5 * S, "keep", 10)


@pytest.mark.parametrize("kwargs", [
    dict(symbols=()),
    dict(symbols=("BTC/USD", "BTC/USD")),
    dict(order_qty={}),
    dict(order_qty={"BTC/USD": "0"}),
    dict(order_qty={"BTC/USD": "abc"}),
    dict(order_qty={"BTC/USD": "1", "ETH/USD": "1"}),
    dict(horizon_ns=0),
    dict(horizon_ns=1.5),
    dict(latency_ns=-1),
    dict(gap_ns=True),
    dict(view_depth=0),
    dict(increase_policy="front"),
    dict(tick_size={"BTC/USD": "-0.1"}),
])
def test_config_rejects_invalid(kwargs):
    base = dict(symbols=("BTC/USD",), order_qty={"BTC/USD": "1"}, horizon_ns=S)
    with pytest.raises(ValueError):
        SimConfig(**{**base, **kwargs})


# --- SimL3Book accessors and BookView -----------------------------------------


def book(depth=10):
    sealer = Sealer(BTC, depth)
    b = SimL3Book(BTC, depth)
    b.apply("snapshot", sealer.snapshot(
        bids=[order("B1", "100.0", "1"), order("B2", "100.0", "2"), order("B3", "99.5", "5")],
        asks=[order("A1", "101.0", "3"), order("A2", "101.5", "1")],
    ))
    return b, sealer


def test_level_rank_counts_better_levels():
    b, _ = book()
    assert [b.level_rank("bids", D(p)) for p in ("100.5", "100.0", "99.7", "99.5", "99.0")] == [1, 1, 2, 2, 3]
    assert [b.level_rank("asks", D(p)) for p in ("100.5", "101.0", "101.2", "101.5", "102.0")] == [1, 1, 2, 2, 3]


def test_level_visible_and_touch_qty():
    b, _ = book(depth=2)
    assert b.level_visible("bids", D("99.5")) and b.level_visible("bids", D("99.7"))
    assert not b.level_visible("bids", D("99.4"))  # worse than the worst retained level, side full
    assert b.level_visible("asks", D("101.5")) and not b.level_visible("asks", D("101.6"))
    assert b.touch_qty("bids") == D("3") and b.touch_qty("asks") == D("3")
    roomy, _ = book(depth=10)
    assert roomy.level_visible("bids", D("1"))  # fewer than depth levels: everything visible
    assert SimL3Book(BTC, 10).touch_qty("bids") == 0


def test_book_view_is_a_frozen_copy():
    b, sealer = book()
    view = b.book_view(123, D("0.1"), 1)
    assert view == BookView("BTC/USD", 123, D("0.1"), (LevelSummary(D("100.0"), D("3"), 2),),
                            (LevelSummary(D("101.0"), D("3"), 1),), b.timestamp)
    assert (view.best_bid, view.best_ask, view.best(Side.BUY), view.best(Side.SELL)) == (
        D("100.0"), D("101.0"), D("100.0"), D("101.0"))
    b.apply("update", sealer.update(bids=[order("B9", "100.5", "1", "add")]))
    assert view.best_bid == D("100.0")  # unchanged by later items
    with pytest.raises(AttributeError):
        view.ts_ns = 0


# --- JoinBest and schedule ---------------------------------------------------------


def test_join_best_joins_own_side_touch():
    view = BookView("BTC/USD", 0, D("0.1"), (LevelSummary(D("100.0"), D("3"), 2),),
                    (LevelSummary(D("101.0"), D("3"), 1),))
    s = JoinBestStrategy()
    assert s.decide(PlacementRequest("r", "BTC/USD", Side.BUY, 0), view) == OrderIntent(Side.BUY, D("100.0"))
    assert s.decide(PlacementRequest("r", "BTC/USD", Side.SELL, 0), view) == OrderIntent(Side.SELL, D("101.0"))
    one_sided = BookView("BTC/USD", 0, D("0.1"), (), view.asks)
    assert s.decide(PlacementRequest("r", "BTC/USD", Side.BUY, 0), one_sided) == Skip(SkipReason.NO_TOUCH)
    assert s.decide(PlacementRequest("r", "BTC/USD", Side.SELL, 0), one_sided) == OrderIntent(Side.SELL, D("101.0"))


def test_regular_schedule():
    reqs = PlacementSchedule.regular(["BTC/USD"], 0, 3 * S, S)
    assert [(r.ts_ns, r.side) for r in reqs] == [(0, Side.BUY), (0, Side.SELL), (S, Side.BUY), (S, Side.SELL),
                                                 (2 * S, Side.BUY), (2 * S, Side.SELL)]
    assert len({r.request_id for r in reqs}) == 6
    assert PlacementSchedule.regular(["BTC/USD"], 0, 3 * S, S) == reqs  # deterministic
    with pytest.raises(ValueError):
        PlacementSchedule.regular(["BTC/USD"], 0, S, 0)
