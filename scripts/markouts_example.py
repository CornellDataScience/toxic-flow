"""Markouts for a list of passive fills, using BookQuery.mid_after.

    python scripts/markouts_example.py                       # recorded trades as stand-in fills
    python scripts/markouts_example.py --fills fills.csv     # your own: columns ts, side, price

A markout asks: after a passive (resting) order was filled, where did the mid
go? For the passive buyer it's mid(t + h) - fill price; for the passive
seller, fill price - mid(t + h). Positive means the fill was profitable at
that horizon. Consistently negative means the orders were filled by informed
("toxic") flow, which is adverse selection.

With no --fills file, every recorded trade is treated as a fill of the resting
order it hit: a buy aggressor fills a passive seller, and vice versa. `side`
in a fills file is the passive order's side (bid = we bought).

Times are exchange-clock ns (or ISO strings). Fills outside valid intervals,
or whose horizon crosses a gap, get NaN.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # import toxicflow without pip install -e

from toxicflow.bookquery import BookQuery, build_tables
from toxicflow.bookquery.units import to_ns_array

HORIZONS = ("100ms", "1s", "10s")


def markouts(bq: BookQuery, fills: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """fills: ts (ns), side ("bid" = passive buy, "ask" = passive sell), price (float).

    Returns one row per fill: markout per horizon in price units and in basis
    points of the fill price.
    """
    ts = to_ns_array(fills["ts"])
    mids = bq.mid_after(ts, horizons).reset_index(drop=True)
    sign = np.where(fills["side"].str.lower().isin(["bid", "buy"]).to_numpy(), 1.0, -1.0)
    price = fills["price"].to_numpy(dtype=float)
    out = pd.DataFrame({"ts": ts, "side": fills["side"].to_numpy(), "price": price, "mid": mids["mid"]})
    for h in horizons:
        out[f"mo_{h}"] = sign * (mids[str(h)].to_numpy() - price)
        out[f"mo_{h}_bps"] = out[f"mo_{h}"] / price * 1e4
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--l3", default="data/raw/l3/*/*.jsonl.gz")
    parser.add_argument("--public", default="data/raw/public/*/*.jsonl.gz")
    parser.add_argument("--fills", type=Path, help="CSV with columns ts, side, price")
    args = parser.parse_args()

    tables = build_tables(glob.glob(args.l3), glob.glob(args.public), symbol=args.symbol)
    bq = BookQuery.load(tables.events, tables.trades, tables.valid_intervals, pair=tables.pair, depth=tables.depth)

    if args.fills:
        fills = pd.read_csv(args.fills)
    else:
        tr = bq.trades
        fills = pd.DataFrame({
            "ts": tr["ts_exchange"],
            "side": np.where(tr["side"] == "buy", "ask", "bid"),  # the resting side that got hit
            "price": bq.price_to_float(tr["price_ticks"].to_numpy()),
        })

    mo = markouts(bq, fills)
    pd.set_option("display.width", 160)
    print(mo.head(10).to_string(index=False))
    cols = [f"mo_{h}_bps" for h in HORIZONS]
    print(f"\n{len(mo)} fills, {mo['mid'].notna().sum()} with a valid book at fill time")
    print("mean markout (bps) by passive side:")
    print(mo.groupby("side")[cols].agg(["mean", "count"]).round(3).to_string())


if __name__ == "__main__":
    main()
