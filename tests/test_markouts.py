import numpy as np
import pandas as pd

from analysis.markouts import compute_markouts, markout_curve


def test_signs_and_horizons():
    mids = pd.DataFrame({"ts": [0, 0.5, 2, 20], "mid": [100.0, 101.0, 102.0, 102.0]})
    fills = pd.DataFrame({"ts": [0, 0], "side": ["bid", "ask"],
                          "price": [99.5, 100.5], "qty": [1, 1]})
    m = compute_markouts(fills, mids).set_index("side")
    # bought at 99.5; mid at 100ms is still 100 -> +0.5 (half spread earned)
    assert m.loc["bid", "markout_100ms"] == 0.5
    # sold at 100.5; mid at 1s is 101 -> -0.5 (adverse)
    assert m.loc["ask", "markout_1s"] == -0.5
    assert m.loc["ask", "markout_10s"] == -1.5


def test_fills_near_end_get_nan():
    mids = pd.DataFrame({"ts": [0, 5], "mid": [100.0, 100.0]})
    fills = pd.DataFrame({"ts": [4], "side": ["bid"], "price": [99.5], "qty": [1]})
    m = compute_markouts(fills, mids)
    assert m["markout_100ms"].notna().all() and np.isnan(m["markout_10s"].iloc[0])


def test_curve_is_volume_weighted():
    m = pd.DataFrame({"qty": [1, 3], "markout_1s": [1.0, -1.0]})
    c = markout_curve(m, horizons=(1.0,))
    assert c["mean_markout"].iloc[0] == -0.5
