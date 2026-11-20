"""Figures for markout_curve: the markout curve, its distribution, and loss concentration.

Each function draws on a new matplotlib Figure and returns it; the caller saves it. Uses the
non-interactive Agg backend, so it works without a display.
"""

from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .markout_curve import Breakeven, horizon_label, lorenz  # noqa: E402
from .timeutil import MS  # noqa: E402

PALETTE = ("#2563eb", "#ea580c", "#16a34a", "#9333ea")
Z95 = 1.96


def _colors(models: Sequence[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(models)}


def _log_time_axis(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: horizon_label(int(round(v * MS)))))
    ax.set_xlabel("time after fill (log scale)")
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.1)


def plot_curve(curve: pd.DataFrame, breakevens: dict[str, Breakeven], models: Sequence[str], title: str) -> Figure:
    """Mean adverse move after the fill (with 95% band) against the -half-spread line, per model.

    Where a model's curve meets its dashed line, the average fill has given back the whole
    spread: the breakeven horizon (dotted vertical line).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = _colors(models)
    for m in models:
        g = curve[curve["model"] == m].sort_values("horizon_ns")
        if g.empty:
            continue
        x = g["horizon_ns"].to_numpy() / MS
        y, se = g["adverse_bps"].to_numpy(), g["adverse_se"].to_numpy()
        c = colors[m]
        ax.plot(x, y, "o-", color=c, lw=2, ms=4, label=f"{m}: mean mid move after fill (n={int(g['n_fills'].iloc[0])})")
        ax.fill_between(x, y - Z95 * np.nan_to_num(se), y + Z95 * np.nan_to_num(se), color=c, alpha=0.15)
        half = g["spread_bps"].iloc[0]
        ax.axhline(-half, color=c, ls="--", lw=1.5, label=f"{m}: -half-spread earned at fill ({half:.4f} bps)")
        b = breakevens.get(m)
        if b is not None and b.status == "crossed":
            bx = b.horizon_ns / MS
            ax.axvline(bx, color=c, ls=":", lw=1.2)
            ax.annotate(f"breakeven {b.label}", (bx, -half), textcoords="offset points", xytext=(6, 8), color=c,
                        fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    _log_time_axis(ax)
    ax.set_ylabel("bps, signed by side (+ = mid moved in our favour)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower left")
    first = curve["horizon"].iloc[0] if len(curve) else ""
    notes = {"before_grid": f"< {first} (already past the line at the first point)", "never": "not reached"}
    lines = [f"{m}: {notes.get(breakevens[m].status, breakevens[m].label)}" for m in models if m in breakevens]
    if lines:
        ax.text(0.99, 0.98, "breakeven (curve meets its dashed line)\n" + "\n".join(lines), transform=ax.transAxes,
                ha="right", va="top", fontsize=8, bbox={"boxstyle": "round", "fc": "white", "ec": "0.7"})
    fig.tight_layout()
    return fig


def plot_distribution(panel: pd.DataFrame, curve: pd.DataFrame, models: Sequence[str], hist_horizon_ns: int,
                      title: str) -> Figure:
    """Top: per-fill net markout percentiles over time (5-95 and 25-75 bands, median, weighted mean).
    Bottom: histogram of net markouts at one horizon."""
    colors = _colors(models)
    fig, axes = plt.subplots(2, len(models), figsize=(6 * len(models), 8), squeeze=False)
    for j, m in enumerate(models):
        g = curve[curve["model"] == m].sort_values("horizon_ns")
        ax = axes[0, j]
        if not g.empty:
            x = g["horizon_ns"].to_numpy() / MS
            c = colors[m]
            ax.fill_between(x, g["net_p5"], g["net_p95"], color=c, alpha=0.12, label="5th-95th pct")
            ax.fill_between(x, g["net_p25"], g["net_p75"], color=c, alpha=0.3, label="25th-75th pct")
            ax.plot(x, g["net_p50"], color=c, lw=1.5, ls="--", label="median")
            ax.plot(x, g["net_bps"], color=c, lw=2, label="qty-weighted mean")
        ax.axhline(0, color="black", lw=0.8)
        _log_time_axis(ax)
        ax.set_ylabel("net markout per fill, bps")
        ax.set_title(m)
        ax.legend(fontsize=8, loc="best")

        ax = axes[1, j]
        vals = panel[(panel["model"] == m) & (panel["horizon_ns"] == hist_horizon_ns)]["net_bps"].to_numpy()
        if len(vals):
            ax.hist(vals, bins=min(60, max(10, len(vals) // 3)), color=colors[m], alpha=0.8)
            ax.axvline(0, color="black", lw=0.8)
            ax.axvline(vals.mean(), color="black", ls="--", lw=1, label=f"mean {vals.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set_xlabel(f"net markout at {horizon_label(hist_horizon_ns)}, bps")
        ax.set_ylabel("fills")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_concentration(panel: pd.DataFrame, models: Sequence[str], horizons_ns: Sequence[int], title: str) -> Figure:
    """Share of total loss from the worst fills (worst first). The diagonal is losses spread evenly
    over all fills; a curve that rises steeply means a few fills cause most of the loss."""
    colors = _colors(models)
    fig, axes = plt.subplots(1, len(horizons_ns), figsize=(5.5 * len(horizons_ns), 5), squeeze=False)
    for j, d in enumerate(horizons_ns):
        ax = axes[0, j]
        curves = lorenz(panel, d)
        for m in models:
            if m in curves:
                x, y = curves[m]
                ax.plot(x * 100, y * 100, color=colors[m], lw=2, label=m)
        ax.plot([0, 100], [0, 100], color="gray", ls=":", lw=1, label="losses spread evenly")
        ax.set_xlabel("% of fills, worst first")
        ax.set_ylabel("% of total loss")
        ax.set_title(f"at {horizon_label(d)}")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
