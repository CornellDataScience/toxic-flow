"""Run one execution-simulation experiment end to end and write its results (design §11).

    python scripts/run_simulation.py data/raw/l3/2026-10-18/l3_2026-10-18T04.jsonl.gz \
        --trades auto --qty 0.001 --interval-s 10 --horizon-s 10

The path is the library's, not a second implementation:

    PlacementSchedule.regular      the placement requests, fixed from the clock before the run
    metrics.evaluate               TradeIndex + attribute_trades (when trade recordings are given)
                                   -> HistoricalReplayEngine (RecordingReader, SimL3Book, strategies,
                                      QueueTracker, FillRecorder) -> sample_mids -> analyze
    results.ResultWriter           data/sim/<run_id>/{manifest.json, schedule, orders, fills, quality}.jsonl

Every strategy sees the same requests, order size, horizon and latency. Several hourly files are
replayed in time order; a file boundary restarts the book, so orders live across it are censored.

Evidence. Without --trades the run is L3-only: fills are bounds from order-book events alone and
nothing is trade-confirmed. With --trades (public-stream recordings of the same hours, or `auto`
to look for them beside the L3 files) fills are settled with trade evidence.

Crossing rule (--cross-rule, see toxicflow/execsim/execution.py). Default `stop`: once an
opposite-side order rests at or through our price, later events are not counted and the order's
outcome is reported as unknown. `ignore` is the earlier behaviour; `fill_at_cross` additionally
lets the optimistic policy count the crossing arrival.

The run id is derived from the inputs and settings, so the same run lands in the same directory;
rerunning it needs --overwrite and reproduces the .jsonl files byte for byte.
"""

import argparse
import gzip
import platform
import sys
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # import toxicflow without pip install -e

from toxicflow.execsim.config import SimConfig  # noqa: E402
from toxicflow.execsim.execution import CROSS_RULES, POLICY_NAMES, REPORTING_CROSS_RULE  # noqa: E402
from toxicflow.execsim.markouts import DEFAULT_MARKOUT_HORIZONS_NS  # noqa: E402
from toxicflow.execsim.metrics import evaluate, format_report  # noqa: E402
from toxicflow.execsim.models import Side  # noqa: E402
from toxicflow.execsim.queue_tracker import INCREASE_POLICIES  # noqa: E402
from toxicflow.execsim.reader import DEFAULT_GAP_NS, file_hour  # noqa: E402
from toxicflow.execsim.results import (ResultWriter, config_dict, file_info, git_state, quality_summary,  # noqa: E402
                                       report_summary, stable_hash)
from toxicflow.execsim.schedule import PlacementSchedule  # noqa: E402
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy  # noqa: E402
from toxicflow.execsim.timeutil import MS, S, parse_rfc3339_ns  # noqa: E402
from toxicflow.execsim.trades import DEFAULT_TOLERANCE  # noqa: E402
from toxicflow.kraken.config import DATA_DIR  # noqa: E402
from toxicflow.kraken.ws import loads  # noqa: E402

STRATEGY_NAMES = ("join_best", "improve_quote", "quote_deeper")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("l3", nargs="+", type=Path, help="L3 recording(s), l3_<date>T<HH>.jsonl.gz; replayed in hour order")
    ap.add_argument("--trades", nargs="*", default=None, metavar="FILE",
                    help="public-stream recording(s) of the same hours, or the word 'auto' to use "
                         "<raw>/public/<date>/public_<date>T<HH>.jsonl.gz beside each L3 file")
    ap.add_argument("--evidence", choices=("auto", "l3_only", "trades"), default="auto",
                    help="auto = trades when trade recordings are given, otherwise l3_only")
    ap.add_argument("--cross-rule", choices=CROSS_RULES, default=REPORTING_CROSS_RULE)
    ap.add_argument("--symbols", nargs="+", default=None, help="default: every pair in the first file's meta line")
    ap.add_argument("--strategies", nargs="+", choices=STRATEGY_NAMES, default=list(STRATEGY_NAMES))
    ap.add_argument("--depth-ticks", type=int, default=1, help="quote_deeper: ticks behind the own-side touch")
    ap.add_argument("--qty", nargs="+", required=True, metavar="QTY | SYMBOL=QTY",
                    help="fixed order size: one number for every symbol, or SYMBOL=QTY per symbol")
    ap.add_argument("--interval-s", type=float, default=10.0, help="spacing of placement requests")
    ap.add_argument("--start", default=None, help="first request time (RFC 3339); default: hour of the first file")
    ap.add_argument("--end", default=None, help="end of the schedule, exclusive; default: end of the last file's hour")
    ap.add_argument("--sides", nargs="+", choices=("buy", "sell"), default=["buy", "sell"])
    ap.add_argument("--horizon-s", type=float, default=10.0, help="holding horizon before the order is canceled")
    ap.add_argument("--latency-ms", type=float, default=0.0, help="decision to insertion delay")
    ap.add_argument("--warmup-s", type=float, default=0.0, help="synced time required before deciding")
    ap.add_argument("--gap-s", type=float, default=DEFAULT_GAP_NS / S, help="receive gap that censors")
    ap.add_argument("--tick", nargs="*", default=[], metavar="SYMBOL=TICK",
                    help="tick size override; default 10**-price_precision from the recording")
    ap.add_argument("--increase-policy", choices=INCREASE_POLICIES, default="keep")
    ap.add_argument("--view-depth", type=int, default=10, help="book levels per side shown to strategies")
    ap.add_argument("--markout-s", nargs="+", type=float, default=[d / S for d in DEFAULT_MARKOUT_HORIZONS_NS],
                    help="markout horizons in seconds")
    ap.add_argument("--audit-invariants", action="store_true", help="check the queue invariant after every item (slow)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "sim", help="directory that holds run directories")
    ap.add_argument("--run-id", default=None, help="default: derived from the inputs and settings (deterministic)")
    ap.add_argument("--overwrite", action="store_true", help="replace the result files of an existing run id")
    args = ap.parse_args(argv)
    if args.evidence == "trades" and not args.trades:
        ap.error("--evidence trades needs --trades (public-stream recordings, or 'auto')")
    return args


def _ns(seconds: float, unit: int = S) -> int:
    return int(round(seconds * unit))


def _pairs(items, what: str) -> dict[str, str]:
    out = {}
    for item in items:
        sym, sep, value = item.rpartition("=")
        if not sep or not sym:
            raise SystemExit(f"{what}: expected SYMBOL=VALUE, got {item!r}")
        out[sym] = value
    return out


def _sorted_by_hour(files: list[Path], stream: str) -> list[Path]:
    """Hour order when every name is <stream>_<date>T<HH>.jsonl.gz; otherwise the order given."""
    try:
        return sorted(files, key=lambda p: file_hour(p, stream))
    except ValueError:
        return list(files)


def _meta_pairs(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        first = loads(fh.readline())
    return list(first.get("meta", {}).get("pairs", {}))


def _trade_files(args, l3: list[Path]) -> list[Path] | None:
    if not args.trades or args.evidence == "l3_only":
        return None
    if args.trades == ["auto"]:
        found = []
        for p in l3:
            hour = file_hour(p, "l3").strftime("%Y-%m-%dT%H")
            found.append(p.parents[2] / "public" / hour[:10] / f"public_{hour}.jsonl.gz")
    else:
        found = [Path(t) for t in args.trades]
    missing = [str(p) for p in found if not p.exists()]
    if missing:
        raise SystemExit(f"trade recording(s) not found: {missing}. Run without --trades for L3-only evidence.")
    return _sorted_by_hour(found, "public")


def _strategies(args) -> tuple[list, list[dict]]:
    built = {"join_best": JoinBestStrategy, "improve_quote": ImproveQuoteStrategy,
             "quote_deeper": lambda: QuoteDeeperStrategy(args.depth_ticks)}
    names = list(dict.fromkeys(args.strategies))  # keep the order given, drop repeats
    strategies = [built[n]() for n in names]
    described = [{"name": s.name, "class": type(s).__name__,
                  "params": {"depth_ticks": s.depth_ticks} if isinstance(s, QuoteDeeperStrategy) else {}}
                 for s in strategies]
    return strategies, described


def main(argv=None) -> Path:
    args = parse_args(argv)
    t0 = time.perf_counter()
    started = datetime.now(timezone.utc)

    l3 = _sorted_by_hour(args.l3, "l3")
    missing = [str(p) for p in l3 if not p.exists()]
    if missing:
        raise SystemExit(f"L3 recording(s) not found: {missing}")
    trade_files = _trade_files(args, l3)
    evidence = "trades" if trade_files else "l3_only"

    symbols = tuple(args.symbols or _meta_pairs(l3[0]))
    if not symbols:
        raise SystemExit("no symbols: the first file has no meta line; pass --symbols")
    qty = {s: args.qty[0] for s in symbols} if len(args.qty) == 1 and "=" not in args.qty[0] else _pairs(
        args.qty, "--qty")
    cfg = SimConfig(symbols=symbols, order_qty=qty, horizon_ns=_ns(args.horizon_s),
                    latency_ns=_ns(args.latency_ms, MS), tick_size=_pairs(args.tick, "--tick"),
                    gap_ns=_ns(args.gap_s), warmup_ns=_ns(args.warmup_s), increase_policy=args.increase_policy,
                    view_depth=args.view_depth, audit_invariants=args.audit_invariants)

    try:
        start_ns = parse_rfc3339_ns(args.start) if args.start else int(file_hour(l3[0], "l3").timestamp()) * S
        end_ns = parse_rfc3339_ns(args.end) if args.end else int(
            (file_hour(l3[-1], "l3") + timedelta(hours=1)).timestamp()) * S
    except ValueError as exc:
        raise SystemExit(f"cannot set the schedule range ({exc}); pass --start and --end as RFC 3339 times")
    sides = tuple(Side(s) for s in dict.fromkeys(args.sides))
    spacing_ns = _ns(args.interval_s)
    schedule = PlacementSchedule.regular(symbols, start_ns, end_ns, spacing_ns, sides)  # from the clock only (§11.4)
    strategies, described = _strategies(args)
    markouts = tuple(_ns(d) for d in args.markout_s)

    inputs = {"l3": [file_info(p) for p in l3], "trades": [file_info(p) for p in trade_files or ()]}
    config = config_dict(cfg)
    run = {"strategies": described, "policies": list(POLICY_NAMES), "evidence_mode": evidence,
           "cross_rule": args.cross_rule, "markout_horizons_ns": list(markouts),
           "trade_tolerance_ns": {"before": DEFAULT_TOLERANCE.before_ns, "after": DEFAULT_TOLERANCE.after_ns}
           if trade_files else None}
    sched = {"kind": "regular", "start_ns": start_ns, "end_ns": end_ns, "spacing_ns": spacing_ns,
             "sides": [s.value for s in sides], "n_requests": len(schedule),
             "hash": stable_hash([[r.request_id, r.symbol, r.side, r.ts_ns] for r in schedule])}
    identity = {"config": stable_hash(config), "run": stable_hash(run), "schedule": sched["hash"],
                "inputs": [[f["name"], f["sha256"]] for group in inputs.values() for f in group]}
    run_hash = stable_hash(identity)
    run_id = args.run_id or f"{l3[0].name.removeprefix('l3_').removesuffix('.jsonl.gz')}_{evidence}_" \
                            f"{args.cross_rule}_{run_hash[:10]}"

    try:
        writer = ResultWriter(args.out / run_id, run_id, overwrite=args.overwrite)
    except FileExistsError as exc:
        raise SystemExit(f"{exc} (--overwrite replaces it)")
    writer.write_schedule(schedule)  # before the replay, so the schedule cannot depend on any outcome

    result, report = evaluate(cfg, strategies, schedule, l3, markout_horizons_ns=markouts, trade_files=trade_files,
                              evidence=evidence, cross_rule=args.cross_rule)
    writer.write(result, report, cfg)
    quality = quality_summary(result)
    code = git_state(REPO)
    writer.write_manifest({
        "created_by": "scripts/run_simulation.py",
        "run_hash": run_hash,
        "code": code,
        "config": config, "config_hash": identity["config"],
        "run": run, "schedule": sched, "inputs": inputs,
        "time_range": {"schedule_start_ns": start_ns, "schedule_end_ns": end_ns,
                       "first_request_ts_ns": schedule[0].ts_ns if schedule else None,
                       "last_request_ts_ns": schedule[-1].ts_ns if schedule else None,
                       "horizon_ns": cfg.horizon_ns},
        "quality": quality,
        "summary": report_summary(report),
        "trade_attribution": report.attribution,
        "runtime": {"started_utc": started.isoformat(timespec="seconds"),
                    "wall_seconds": round(time.perf_counter() - t0, 3),
                    "python": platform.python_version(), "platform": platform.platform(),
                    "argv": [str(a) for a in (sys.argv[1:] if argv is None else argv)]},
    })

    shown = tuple(d for d in markouts if d in (S, 5 * S, 10 * S)) or markouts
    label = ("L3 + trade prints" if evidence == "trades" else
             "L3-ONLY: no trade recordings were used, so no fill here is trade-confirmed")
    print(f"run {run_id}\nevidence: {label}\ninputs: {', '.join(p.name for p in l3)}"
          + (f" + {', '.join(p.name for p in trade_files)}" if trade_files else ""))
    print(format_report(report, markout_horizons_ns=shown))
    print("\ndata quality (records by kind; one incident can produce several records)")
    for sym in symbols:
        print(f"  {sym}: incidents {quality['incidents'].get(sym, 0)}, {quality['events_by_kind'].get(sym, {})}")
    if code["dirty"]:
        print(f"\nnote: the working tree has {len(code['changed_files'])} uncommitted change(s); the manifest "
              f"records commit {str(code['commit'])[:7]} as dirty")
    print("\nrows: " + ", ".join(f"{name} {info['rows']}" for name, info in writer.files.items()))
    print(f"output: {writer.run_dir}")
    return writer.run_dir


if __name__ == "__main__":
    main()
