"""Markout curve analysis: decomposition, breakeven, concentration and the naive fill model."""

import math
from decimal import Decimal as D

import numpy as np
import pandas as pd
import pytest

from toxicflow.execsim import markout_curve as mc
from toxicflow.execsim.models import Side
from toxicflow.execsim.timeutil import MS, S
from toxicflow.execsim.trades import Trade
from toxicflow.kraken.config import DATA_DIR


def trade(ts, taker, price, qty="1"):
    return Trade("X/USD", Side(taker), D(price), D(qty), None, ts, "limit", (0, 0))


def fills_df(rows):
    """rows: (fill_id, side, ts_ns, price, qty)."""
    out = []
    for fid, side, ts, price, qty in rows:
        sign = 1 if side == "buy" else -1
        out.append(("m", fid, fid, "X/USD", side, sign, ts, D(price), D(qty), 0, D(0), 0, "trade", D(1), False))
    return pd.DataFrame(out, columns=mc.FILL_COLUMNS)


# --- horizons -------------------------------------------------------------------------------------------


def test_log_horizons_default_grid():
    g = mc.log_horizons()
    assert g[0] == 10 * MS and g[-1] == 10 * S and len(g) == 19
    ratios = np.diff(np.log10(g))
    assert np.allclose(ratios, 1 / 6, atol=1e-6)
    assert mc.horizon_label(10 * MS) == "10ms" and mc.horizon_label(10 * S) == "10s"


# --- naive fills ---------------------------------------------------------------------------------------------


def naive(side, price, trades, start=100, stop=200, **cross):
    s = Side(side)
    return mc.naive_touch(s.sign, D(price), s.book_side, start, stop, trades, [t.ts_ns for t in trades], **cross)


def test_naive_buy_touched_by_sell_print_at_or_below_price():
    trades = [trade(110, "sell", "100.5"),  # above our bid: does not reach us
              trade(120, "buy", "99"),      # buy aggressor hits asks, not our bid
              trade(130, "sell", "100")]    # at our price
    t = naive("buy", "100", trades)
    assert (t.ts_ns, t.trigger, t.through) == (130, "trade", False)
    t = naive("buy", "100", [trade(130, "sell", "99.5")])
    assert t.through  # traded through our price


def test_naive_sell_and_window_bounds():
    trades = [trade(100, "buy", "101"), trade(200, "buy", "101")]
    assert naive("sell", "101", trades).ts_ns == 100           # start is inclusive
    assert naive("sell", "101", trades[1:]) is None            # stop is exclusive
    assert naive("sell", "101", [trade(150, "buy", "100.5")]) is None  # below our ask


def test_naive_crossing_counts_as_touch_when_earlier():
    trades = [trade(150, "sell", "100")]
    t = naive("buy", "100", trades, cross_ts=140, cross_price=D("99.5"), cross_qty=D("2"))
    assert (t.ts_ns, t.trigger, t.event_qty, t.through) == (140, "cross", D("2"), True)
    assert naive("buy", "100", trades, cross_ts=160).trigger == "trade"
    assert naive("buy", "100", [], cross_ts=250) is None       # outside the window


# --- mid with our order, decomposition ----------------------------------------------------------------------


def test_mid_with_order_puts_our_order_at_the_touch():
    q = mc.Quote(D("100"), D("101"), None)
    assert mc.mid_with_order(q, +1, D("100")) == (D("100.5"), None)
    # the real bid queue at our price emptied: the recorded bid is 99.5, ours is still 100
    assert mc.mid_with_order(mc.Quote(D("99.5"), D("101"), None), +1, D("100")) == (D("100.5"), None)
    assert mc.mid_with_order(mc.Quote(D("100"), D("102"), None), -1, D("101")) == (D("100.5"), None)
    assert mc.mid_with_order(mc.Quote(D("100"), D("101"), None), +1, D("101"))[1] == "CROSSED_WITH_ORDER"
    assert mc.mid_with_order(mc.Quote(None, None, "DATA_GAP"), +1, D("100")) == (None, "DATA_GAP")


def test_decompose_known_values_and_additivity():
    fills = fills_df([("b", "buy", 1000, "100", "2"), ("s", "sell", 1000, "101", "1")])
    quotes = {("X/USD", 1000): mc.Quote(D("100"), D("101"), None),
              ("X/USD", 1010): mc.Quote(D("99"), D("100"), None),     # mid 99.5: bad for the buy, good for the sell
              ("X/USD", 1100): mc.Quote(D("100"), D("101"), None)}
    long = mc.decompose(fills, quotes, (10, 100)).set_index(["fill_id", "horizon_ns"])
    b = long.loc[("b", 10)]
    assert b["spread_bps"] == pytest.approx(0.5 / 100.5 * 1e4)
    assert b["adverse_bps"] == pytest.approx(-1.0 / 100.5 * 1e4)
    assert b["net_usd"] == pytest.approx(-0.5) and b["pnl_usd"] == pytest.approx(-1.0)  # qty 2
    s = long.loc[("s", 10)]
    assert s["spread_bps"] == pytest.approx(0.5 / 100.5 * 1e4) and s["adverse_bps"] == pytest.approx(1 / 100.5 * 1e4)
    assert np.allclose(long["spread_bps"] + long["adverse_bps"], long["net_bps"])
    assert long.loc[("b", 100), "adverse_bps"] == pytest.approx(0)


def test_missing_quotes_are_nan_and_dropped_from_the_panel():
    fills = fills_df([("a", "buy", 1000, "100", "1"), ("b", "buy", 2000, "100", "1")])
    quotes = {("X/USD", 1000): mc.Quote(D("100"), D("101"), None), ("X/USD", 1010): mc.Quote(D("100"), D("101"), None),
              ("X/USD", 1100): mc.Quote(D("100"), D("101"), None),
              ("X/USD", 2000): mc.Quote(D("100"), D("101"), None), ("X/USD", 2010): mc.Quote(D("100"), D("101"), None),
              ("X/USD", 2100): mc.Quote(None, None, "DATA_GAP")}
    long = mc.decompose(fills, quotes, (10, 100))
    assert long.set_index(["fill_id", "horizon_ns"]).loc[("b", 100), "missing"] == "DATA_GAP"
    panel, dropped = mc.complete_panel(long)
    assert set(panel["fill_id"]) == {"a"} and dropped.to_dict() == {"m": 1}  # b dropped at every horizon


def test_required_times():
    fills = fills_df([("a", "buy", 1000, "100", "1")])
    assert mc.required_times(fills, (10, 100)) == {("X/USD", 1000), ("X/USD", 1010), ("X/USD", 1100)}


# --- curve statistics -------------------------------------------------------------------------------------------


def panel_df(rows):
    """rows: (model, fill_id, horizon_ns, spread, adverse, qty, ts_ns)."""
    df = pd.DataFrame(rows, columns=["model", "fill_id", "horizon_ns", "spread_bps", "adverse_bps", "qty", "ts_ns"])
    df["net_bps"] = df["spread_bps"] + df["adverse_bps"]
    df["net_usd"] = df["net_bps"] / 100
    df["pnl_usd"] = df["net_usd"] * df["qty"]
    return df


def test_curve_is_qty_weighted_with_clustered_se():
    p = panel_df([("m", "a", 10, 1.0, -2.0, 3.0, 1), ("m", "b", 10, 1.0, 2.0, 1.0, 1), ("m", "c", 10, 1.0, 0.0, 1.0, 2)])
    row = mc.curve_table(p).iloc[0]
    assert row["adverse_bps"] == pytest.approx((3 * -2 + 2) / 5)
    assert row["n_fills"] == 3 and row["n_events"] == 2
    # two clusters: {a, b} at ts 1 and {c} at ts 2
    m = -0.8
    r1, r2 = 3 * (-2 - m) + 1 * (2 - m), 1 * (0 - m)
    assert row["adverse_se"] == pytest.approx(math.sqrt((r1 ** 2 + r2 ** 2) * 2) / 5)
    assert row["frac_losing"] == pytest.approx(1 / 3)


def curve_from(nets, model="m"):
    ds = (10 * MS, 100 * MS, 1000 * MS)[: len(nets)]
    return pd.DataFrame({"model": model, "horizon_ns": ds, "net_bps": nets})


def test_breakeven_interpolates_in_log_time():
    b = mc.breakeven(curve_from([0.5, 0.1, -0.3]))["m"]
    assert b.status == "crossed"
    assert b.horizon_ns == pytest.approx(10 ** (8 + 0.25), rel=1e-9)  # 1/4 of the way from 100 ms to 1 s
    assert mc.breakeven(curve_from([-0.1, -0.2, -0.3]))["m"].status == "before_grid"
    assert mc.breakeven(curve_from([0.3, 0.2, 0.1]))["m"].status == "never"
    assert mc.breakeven(curve_from([0.3, 0.0, -0.1]))["m"].horizon_ns == pytest.approx(100 * MS)


def test_pnl_split_rows():
    p = panel_df([("m", "a", 10, 2.0, -0.5, 1.0, 1), ("m", "a", 100, 2.0, -3.0, 1.0, 1)])
    rows = mc.pnl_split(mc.curve_table(p), [10, 100])
    assert list(rows["net_bps"]) == pytest.approx([1.5, -1.0])
    assert list(rows["fee_bps"]) == [0.0, 0.0]
    assert list(rows["spread_kept"]) == pytest.approx([0.75, -0.5])


def test_concentration_shares():
    pnl = [-6, -3, -1, 2, 4] + [0] * 5  # 10 fills, total loss 10, net -4
    p = panel_df([("m", f"f{i}", 10, 0.0, v, 1.0, i) for i, v in enumerate(pnl)])
    p["pnl_usd"] = pnl
    row = mc.concentration(p, 10, shares=(0.1, 0.2, 0.5)).iloc[0]
    assert row["worst_10%_loss_share"] == pytest.approx(0.6)
    assert row["worst_20%_loss_share"] == pytest.approx(0.9)
    assert row["worst_50%_loss_share"] == pytest.approx(1.0)
    assert row["worst_10%_net_share"] == pytest.approx(1.5)  # the worst fill lost more than the total net loss
    assert row["frac_losing"] == pytest.approx(0.3)
    x, y = mc.lorenz(p, 10)["m"]
    assert x[0] == 0 and y[-1] == pytest.approx(1.0) and np.all(np.diff(y) >= 0)


def test_worst_profile_splits_groups():
    p = panel_df([("m", f"f{i}", 10, 0.0, float(i), 1.0, i) for i in range(10)])
    for c, v in (("since_insert_ns", 0), ("queue_ahead_qty0", D(0)), ("queue_ahead_orders0", 0), ("event_qty", D(1)),
                 ("side", "buy"), ("through", False), ("trigger", "trade")):
        p[c] = v
    prof = mc.worst_profile(p, 10, 0.1).set_index("group")
    assert prof.loc["worst", "n_fills"] == 1 and prof.loc["rest", "n_fills"] == 9
    assert prof.loc["worst", "net_bps"] == 0.0


# --- end to end on a recording ------------------------------------------------------------------------------


L3_FILES = sorted((DATA_DIR / "raw" / "l3").glob("*/l3_*.jsonl.gz"))
PUB_FILES = sorted((DATA_DIR / "raw" / "public").glob("*/public_*.jsonl.gz"))


@pytest.mark.skipif(not (L3_FILES and PUB_FILES), reason="needs L3 and public recordings under data/raw")
def test_report_end_to_end(tmp_path):
    from scripts.markout_report import main

    path = min(L3_FILES, key=lambda p: p.stat().st_size)
    out = main([str(path), "--trades", "auto", "--qty", "0.001", "--interval-s", "5", "--out", str(tmp_path)])
    for name in ("markout_curve.png", "markout_distribution.png", "loss_concentration.png", "curve.csv",
                 "pnl_split.csv", "concentration.csv", "worst_fills_profile.csv", "fill_rates.csv", "fills.csv",
                 "summary.md"):
        assert (out / name).exists(), name
    long = pd.read_csv(out / "fills.csv")
    ok = long.dropna(subset=["net_bps"])
    assert len(ok) and (ok["spread_bps"] >= 0).all()  # a resting order never pays the spread
    assert np.allclose(ok["spread_bps"] + ok["adverse_bps"], ok["net_bps"])
    rates = pd.read_csv(out / "fill_rates.csv").set_index("model")
    assert rates.loc["naive", "orders"] == rates.loc["queue_aware:balanced", "orders"]  # same orders
