"""Example: replay one recorded L3 hour with the three strategies and print the comparison.

    python scripts/compare_strategies.py [path/to/l3_<date>T<HH>.jsonl.gz]

Defaults to the smallest recording under data/raw/l3. When the public-stream recording of the
same hour exists (data/raw/public/<date>/public_<date>T<HH>.jsonl.gz), fills are also settled
with trade evidence and both reports are printed. Step by step, evaluate_modes() is:

    index = TradeIndex.from_files(trade_files, cfg.symbols, gap_ns=cfg.gap_ns)   # trade prints
    attribution = attribute_trades(cfg, index, files=files)                      # label every reduction
    result = HistoricalReplayEngine(cfg, strategies, schedule,
                                    on_reach=FillRecorder(attribution)).run(files)  # 1. replay
    mids = sample_mids(cfg, required_mid_times(result, cfg, evidence=modes), files=files)
    report = analyze(result, cfg, mids, evidence="trades", attribution=attribution)  # 2. analyze
    rows = comparison_rows(report, "balanced")                                   # 3. compare
    print(format_report(report))                                                 # 4. summary
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toxicflow.execsim.config import SimConfig  # noqa: E402
from toxicflow.execsim.execution import REPORTING_CROSS_RULE  # noqa: E402
from toxicflow.execsim.metrics import comparison_rows, evaluate_modes, format_report  # noqa: E402
from toxicflow.execsim.reader import file_hour, list_files  # noqa: E402
from toxicflow.execsim.schedule import PlacementSchedule  # noqa: E402
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy  # noqa: E402
from toxicflow.execsim.timeutil import S  # noqa: E402
from toxicflow.kraken.config import DATA_DIR  # noqa: E402


def main() -> None:
    files = [Path(sys.argv[1])] if len(sys.argv) > 1 else [min(list_files(DATA_DIR / "raw", "l3"),
                                                                key=lambda p: p.stat().st_size)]
    symbols = ("BTC/USD", "ETH/USD")
    start = int(file_hour(files[0], "l3").timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, start + 3600 * S, 10 * S)  # fixed in advance (§11.4)
    cfg = SimConfig(symbols=symbols, order_qty={s: "0.001" for s in symbols}, horizon_ns=10 * S)
    strategies = [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()]

    hour = file_hour(files[0], "l3").strftime("%Y-%m-%dT%H")
    public = files[0].parents[2] / "public" / hour[:10] / f"public_{hour}.jsonl.gz"
    trade_files = [public] if public.exists() else None
    _, reports = evaluate_modes(cfg, strategies, schedule, files, trade_files=trade_files,
                                cross_rule=REPORTING_CROSS_RULE)  # see scripts/run_simulation.py for saved runs
    for report in reports.values():
        print(f"{files[0].name}\n{format_report(report, markout_horizons_ns=(S, 5 * S, 10 * S))}\n")
    if trade_files is None:
        print(f"no trade recording at {public}: trade-aware evidence unavailable, L3-only bounds only\n")
    report = reports.get("trades", reports["l3_only"])
    for row in comparison_rows(report, "balanced"):
        print({k: (str(v) if v is not None else None) for k, v in row.items() if "markout_1s" in k or k in (
            "strategy", "placement_rate", "fill_given_placed", "fill_per_opportunity", "fill_common_support",
            "median_ahead_qty0", "mean_edge_bps")})


if __name__ == "__main__":
    main()
