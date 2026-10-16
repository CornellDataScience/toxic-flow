"""Markouts: how the market moved against (or for) us after each fill.

markout_h = sign * (mid(t + h) - fill_price)
  sign = +1 if we bought (bid filled), -1 if we sold (ask filled)
  positive = good for us, negative = adverse selection

Measured against the FILL PRICE, so a fill with no subsequent price move
scores +half-spread. That is the spread we "earned"; the drift below it is the
adverse selection cost. Write down whichever convention the team picks.
"""
import numpy as np
import pandas as pd

HORIZONS = (0.1, 1.0, 10.0)  # seconds: 100ms, 1s, 10s


def compute_markouts(fills: pd.DataFrame, mids: pd.DataFrame, horizons=HORIZONS):
    """fills needs columns ts, side, price. mids needs columns ts, mid (sorted)."""
    if fills.empty:
        return fills.copy()
    out = fills.sort_values("ts").reset_index(drop=True).copy()
    out["ts"] = out["ts"].astype(float)
    mids = mids.astype({"ts": float, "mid": float}).sort_values("ts")
    sign = np.where(out["side"] == "bid", 1.0, -1.0)

    at_fill = pd.merge_asof(out[["ts"]], mids, on="ts", direction="backward")
    out["mid_at_fill"] = at_fill["mid"].to_numpy()

    for h in horizons:
        target = pd.DataFrame({"ts": out["ts"] + h})
        fut = pd.merge_asof(target, mids, on="ts", direction="backward")  # last mid known at t+h
        col = f"markout_{_label(h)}"
        out[col] = sign * (fut["mid"].to_numpy() - out["price"].to_numpy())
        out[col + "_bps"] = 1e4 * out[col] / out["mid_at_fill"]
    # Fills too close to the end of the data have no valid future mid
    last = mids["ts"].iloc[-1] if len(mids) else -np.inf
    for h in horizons:
        out.loc[out["ts"] + h > last, [f"markout_{_label(h)}", f"markout_{_label(h)}_bps"]] = np.nan
    return out


def markout_curve(markouts: pd.DataFrame, horizons=HORIZONS, weight="qty"):
    """Volume-weighted average markout per horizon (the curve to plot)."""
    rows = []
    for h in horizons:
        col = f"markout_{_label(h)}"
        d = markouts.dropna(subset=[col])
        w = d[weight] if weight else None
        rows.append({"horizon_s": h, "mean_markout": np.average(d[col], weights=w) if len(d) else np.nan,
                     "n_fills": len(d)})
    return pd.DataFrame(rows)


def label_toxic(markouts: pd.DataFrame, horizon=1.0, threshold=0.0):
    """PLACEHOLDER label for the classifier: toxic if markout at `horizon` < threshold.
    The team still needs to agree on the horizon and threshold."""
    return (markouts[f"markout_{_label(horizon)}"] < threshold).astype(int)


def _label(h):
    return f"{int(h * 1000)}ms" if h < 1 else f"{int(h)}s"
