"""Markout curves for passive fills: queue-aware vs naive, with the PnL split and loss concentration.

    python scripts/markout_report.py data/raw/l3/2026-10-18/l3_2026-10-18T04.jsonl.gz \
        --trades auto --qty 0.001 --interval-s 1

Places the same hypothetical orders as scripts/run_simulation.py (one strategy, default
join_best: rest at the best bid / ask), then compares two ways of deciding when they fill:

    queue_aware  execsim's queue tracker: we join the back of the real queue (headline policy:
                 --policy, default balanced; the tables also show the other policies)
    naive        filled in full the first time the price touches our level (needs --trades)

For each fill: spread earned at the fill, adverse mid move at log-spaced times after it
(--min-ms to --max-s), and net = spread + adverse (toxicflow/execsim/markout_curve.py has the
exact definitions). Writes to --out (default data/reports/markouts/<run>/):

    markout_curve.png         mean adverse move vs the -half-spread line; breakeven where they meet
    markout_distribution.png  per-fill percentiles over time, and a histogram at --hist-s
    loss_concentration.png    share of total loss from the worst X% of fills
    curve.csv                 every model x horizon: means, clustered SEs, percentiles
    pnl_split.csv             spread / adverse selection / fee / net at --split-s horizons
    concentration.csv         worst 1/5/10/20/50% of fills' share of the loss, at --conc-s
    worst_fills_profile.csv   features of the worst 10% of fills vs the rest
    fill_rates.csv            how often each model fills the same orders
    fills.csv                 one row per (fill, horizon), for modeling
    summary.md                the headline numbers
"""

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # import toxicflow without pip install -e

from scripts.run_simulation import _meta_pairs, _pairs, _sorted_by_hour, _trade_files  # noqa: E402
from toxicflow.execsim import markout_curve as mc  # noqa: E402
from toxicflow.execsim.config import SimConfig  # noqa: E402
from toxicflow.execsim.execution import CROSS_RULES, POLICY_NAMES, REPORTING_CROSS_RULE  # noqa: E402
from toxicflow.execsim.markout_plots import plot_concentration, plot_curve, plot_distribution  # noqa: E402
from toxicflow.execsim.metrics import evaluate  # noqa: E402
from toxicflow.execsim.models import Side  # noqa: E402
from toxicflow.execsim.reader import file_hour  # noqa: E402
from toxicflow.execsim.schedule import PlacementSchedule  # noqa: E402
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy  # noqa: E402
from toxicflow.execsim.timeutil import MS, S, format_ns, parse_rfc3339_ns  # noqa: E402
from toxicflow.execsim.trades import TradeIndex  # noqa: E402
from toxicflow.kraken.config import DATA_DIR  # noqa: E402

STRATEGIES = {"join_best": JoinBestStrategy, "improve_quote": ImproveQuoteStrategy, "quote_deeper": QuoteDeeperStrategy}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("l3", nargs="+", type=Path, help="L3 recording(s), l3_<date>T<HH>.jsonl.gz")
    ap.add_argument("--trades", nargs="*", default=None, metavar="FILE",
                    help="public-stream recording(s) of the same hours, or 'auto'. Needed for the naive model "
                         "and for trade-confirmed queue-aware fills")
    ap.add_argument("--qty", nargs="+", required=True, metavar="QTY | SYMBOL=QTY")
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--strategy", choices=tuple(STRATEGIES), default="join_best")
    ap.add_argument("--policy", choices=POLICY_NAMES, default="balanced", help="queue-aware policy in the figures")
    ap.add_argument("--cross-rule", choices=CROSS_RULES, default=REPORTING_CROSS_RULE)
    ap.add_argument("--interval-s", type=float, default=1.0, help="spacing of placement requests")
    ap.add_argument("--start", default=None, help="first request (RFC 3339); default: hour of the first file")
    ap.add_argument("--end", default=None, help="end of the schedule; default: end of the last file's hour")
    ap.add_argument("--horizon-s", type=float, default=10.0, help="how long an order rests before it is canceled")
    ap.add_argument("--latency-ms", type=float, default=0.0)
    ap.add_argument("--min-ms", type=float, default=10.0, help="first markout horizon")
    ap.add_argument("--max-s", type=float, default=10.0, help="last markout horizon")
    ap.add_argument("--per-decade", type=int, default=6, help="markout horizons per factor of 10")
    ap.add_argument("--split-s", nargs="+", type=float, default=[0.01, 0.1, 1.0, 10.0],
                    help="horizons in the PnL split table (snapped to the nearest grid point)")
    ap.add_argument("--conc-s", nargs="+", type=float, default=[1.0, 10.0], help="concentration horizons")
    ap.add_argument("--hist-s", type=float, default=1.0, help="histogram horizon")
    ap.add_argument("--worst-share", type=float, default=0.10, help="worst fraction of fills to profile")
    ap.add_argument("--out", type=Path, default=None, help="default: data/reports/markouts/<run>")
    return ap.parse_args(argv)


def _snap(grid: tuple[int, ...], seconds: list[float]) -> list[int]:
    """Nearest grid horizon (in log time) to each requested one, without repeats."""
    logs = np.log(np.array(grid, float))
    out = [grid[int(np.argmin(np.abs(logs - np.log(s * S))))] for s in seconds]
    return list(dict.fromkeys(out))


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    def cell(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def main(argv=None) -> Path:
    args = parse_args(argv)
    t0 = time.perf_counter()
    l3 = _sorted_by_hour(args.l3, "l3")
    missing = [str(p) for p in l3 if not p.exists()]
    if missing:
        raise SystemExit(f"L3 recording(s) not found: {missing}")
    args.evidence = "auto"
    trade_files = _trade_files(args, l3)
    symbols = tuple(args.symbols or _meta_pairs(l3[0]))
    qty = {s: args.qty[0] for s in symbols} if len(args.qty) == 1 and "=" not in args.qty[0] else _pairs(
        args.qty, "--qty")
    cfg = SimConfig(symbols=symbols, order_qty=qty, horizon_ns=int(args.horizon_s * S),
                    latency_ns=int(args.latency_ms * MS))
    start = parse_rfc3339_ns(args.start) if args.start else int(file_hour(l3[0], "l3").timestamp()) * S
    end = parse_rfc3339_ns(args.end) if args.end else int(
        (file_hour(l3[-1], "l3") + timedelta(hours=1)).timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, end, int(args.interval_s * S), (Side.BUY, Side.SELL))
    strategy = STRATEGIES[args.strategy]()
    grid = mc.log_horizons(int(args.min_ms * MS), int(args.max_s * S), args.per_decade)
    evidence = "trades" if trade_files else "l3_only"

    print(f"replaying {len(l3)} file(s), {len(schedule)} placement requests, evidence={evidence}, "
          f"cross rule={args.cross_rule}")
    result, report = evaluate(cfg, [strategy], schedule, l3, markout_horizons_ns=(grid[-1],),
                              trade_files=trade_files, evidence=evidence, cross_rule=args.cross_rule)

    frames = [mc.queue_aware_fills(report, p, strategy.name) for p in POLICY_NAMES]
    naive = pd.DataFrame(columns=mc.FILL_COLUMNS)
    naive_counts = {"orders": 0, "filled": 0, "by_trade": 0, "by_cross": 0}
    if trade_files:
        index = TradeIndex.from_files(trade_files, symbols, gap_ns=cfg.gap_ns)
        crossings = {o.order_id: (o.cross.price, o.cross.qty) for o in result.orders if o.cross is not None}
        naive, naive_counts = mc.naive_fills(report, index, strategy.name, crossings)
        frames.append(naive)
    else:
        print("no trade recordings: the naive model needs trade prints, so only queue-aware fills are shown")
    fills = pd.concat([f for f in frames if len(f)], ignore_index=True) if any(len(f) for f in frames) else naive
    if fills.empty:
        raise SystemExit("no fills in this data; try more files, a shorter --interval-s or a longer --horizon-s")

    quotes = mc.sample_quotes(cfg, mc.required_times(fills, grid), files=l3)
    long = mc.decompose(fills, quotes, grid)
    panel, dropped = mc.complete_panel(long)
    curve = mc.curve_table(panel)
    be = mc.breakeven(curve)
    split_h = _snap(grid, args.split_s)
    conc_h = _snap(grid, args.conc_s)
    hist_h = _snap(grid, [args.hist_s])[0]
    split = mc.pnl_split(curve, split_h)
    conc = pd.concat([mc.concentration(panel, d) for d in conc_h], ignore_index=True)
    profile = mc.worst_profile(panel, conc_h[0], args.worst_share)
    rates = mc.fill_rates(report, strategy.name, args.policy, naive, naive_counts)

    headline = [f"{mc.QUEUE_AWARE}:{args.policy}"] + ([mc.NAIVE] if trade_files else [])
    headline = [m for m in headline if m in set(curve["model"])]
    sym_label = ", ".join(symbols)
    first, last = (min(r.request.ts_ns for r in report.requests), max(r.request.ts_ns for r in report.requests))
    run = f"{l3[0].name.removeprefix('l3_').removesuffix('.jsonl.gz')}_{strategy.name}_{args.policy}"
    out = args.out or DATA_DIR / "reports" / "markouts" / run
    out.mkdir(parents=True, exist_ok=True)

    title = f"{sym_label} {strategy.name}: markout after passive fills"
    for name, fig in (
        ("markout_curve.png", plot_curve(curve, be, headline, title)),
        ("markout_distribution.png", plot_distribution(panel, curve, headline, hist_h, f"{title}, distribution")),
        ("loss_concentration.png", plot_concentration(panel, headline, conc_h, f"{title}, loss concentration")),
    ):
        fig.savefig(out / name, dpi=150)
    curve.to_csv(out / "curve.csv", index=False)
    split.to_csv(out / "pnl_split.csv", index=False)
    conc.to_csv(out / "concentration.csv", index=False)
    profile.to_csv(out / "worst_fills_profile.csv", index=False)
    rates.to_csv(out / "fill_rates.csv", index=False)
    long.to_csv(out / "fills.csv", index=False)

    eligible = sum(r.status == "ELIGIBLE" for r in report.requests)
    be_rows = pd.DataFrame([{"model": m, "breakeven": be[m].label} for m in headline])
    conc_cols = ["model", "horizon", "n_fills", "frac_losing", "worst_1%_loss_share", "worst_10%_loss_share",
                 "worst_20%_loss_share", "net_pnl_usd"]
    summary = "\n".join([
        f"# Markouts: {sym_label}, {strategy.name}",
        "",
        f"- data: {', '.join(p.name for p in l3)}" + (f" + {', '.join(p.name for p in trade_files)}" if trade_files else ""),
        f"- requests {format_ns(first)} to {format_ns(last)} every {args.interval_s:g}s, both sides; "
        f"{eligible} of {len(report.requests)} eligible; order size {qty}; rests {args.horizon_s:g}s",
        f"- evidence {evidence}, cross rule {args.cross_rule}; markouts {mc.horizon_label(grid[0])} to "
        f"{mc.horizon_label(grid[-1])} ({len(grid)} points)",
        f"- fills dropped from the curve for a missing mid at some horizon: {dropped.to_dict() or 'none'}",
        "",
        "## Breakeven (mean net markout reaches 0)",
        _md(be_rows),
        "## PnL split (bps, qty-weighted; + = good for us)",
        _md(split[split["model"].isin(headline)]),
        "## Fill rates on the same orders",
        _md(rates, "{:.3f}"),
        "## Loss concentration",
        _md(conc[conc["model"].isin(headline)][conc_cols], "{:.3f}"),
        f"## Worst {args.worst_share:.0%} of fills vs the rest at {mc.horizon_label(conc_h[0])}",
        _md(profile[profile["model"].isin(headline)].dropna(axis=1, how="all"), "{:.3f}"),
        "",
        "All policies are in the CSV files next to this summary.",
    ])
    (out / "summary.md").write_text(summary)
    print(summary)
    print(f"\nwrote {out} in {time.perf_counter() - t0:.1f}s")
    return out


if __name__ == "__main__":
    main()
