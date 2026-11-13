"""In-memory replay results, and ResultWriter: one run directory of JSONL files (design §11.1 to §11.3).

    data/sim/<run_id>/
        manifest.json    what was run, on which inputs and code, and summary counts
        schedule.jsonl   the placement requests, written before the replay
        orders.jsonl     one row per (request, strategy): placed, skipped or data-gated
        fills.jsonl      one row per fill CANDIDATE (a historical event that could have reached
                         the order), with its evidence and what each policy accepted of it
        quality.jsonl    discontinuities and data-quality events of the recording

Everything in the four .jsonl files and in manifest.json outside its `runtime` block is a pure
function of the inputs, the configuration and the code, so a rerun is byte-identical there.
Prices and quantities are written as decimal strings, times as integer nanoseconds of exchange
time. Rows are written one at a time; nothing is rebuilt in memory for the output.

Candidates are historical observations and are written unchanged whatever the settlement rule.
What a policy counts (accepted_qty_<policy>, filled_qty_<policy>, ...) is the counterfactual
part and depends on the evidence mode and crossing rule named in the manifest and `fill_model`.
"""

import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from pathlib import Path

from .models import BookView, PlacementRequest, ReachEvent, SkipReason
from .orders import SimulatedOrder


DISCONTINUITY_KINDS = frozenset({"SESSION_RESTART", "RECV_GAP", "TRUNCATED", "DESYNC", "RESNAPSHOT"})


@dataclass(frozen=True, slots=True)
class QualityEvent:
    """A discontinuity or data-quality observation, in stream order.

    One break in the data usually produces several records: a checksum failure is
    DESYNC and, when the book is reloaded, RESNAPSHOT; an hourly file boundary is
    SESSION_RESTART and then RESNAPSHOT. All records are kept for diagnostics, and
    `incident` groups them: a discontinuity raised while the symbol is synced opens
    a new incident, one raised while it is still broken joins the open one, and the
    next sync closes it. Count incidents as distinct (symbol, incident) pairs, never
    as records. Non-discontinuity kinds have incident None.
    """

    ts_ns: int  # the symbol's guarded clock when it was seen
    symbol: str
    kind: str  # DISCONTINUITY_KINDS, SNAPSHOT_FIRST, SNAPSHOT_CHECKSUM_FAIL, TS_REGRESSION, TS_MISSING, ...
    seq: int | None  # StreamRecord.seq where it was seen
    recv_ns: int | None
    detail: dict = field(default_factory=dict)
    incident: int | None = None  # per-symbol incident number (discontinuities only)
    file: str | None = None  # name of the recording file and the line where it was seen
    line_no: int | None = None


@dataclass(frozen=True, slots=True)
class SkipRecord:
    """A (request, strategy) pair for which no SimulatedOrder was created."""

    request: PlacementRequest
    strategy: str
    reason: SkipReason


@dataclass(frozen=True, slots=True)
class ReachRecord:
    """A ReachEvent logged for an order. Not a fill: it carries no evidence label yet.

    Every ReachRecord comes from an item whose checksum passed (I7); events of a failing
    item are never routed, so unvalidated evidence cannot appear here.
    """

    order_id: str
    strategy: str
    symbol: str
    reach: ReachEvent
    ahead_qty: Decimal  # queue ahead when it was seen (0 for LEVEL and CROSS_ARRIVAL)
    checksum_covered: bool  # event price within the checksummed top levels of its side after the item
    source_file: str | None = None  # raw line of the historical event: recording file name and line
    source_line: int | None = None


@dataclass
class ReplayResult:
    orders: list[SimulatedOrder]  # every order created, ended or skipped at insertion, in creation order
    skips: list[SkipRecord]
    reaches: list[ReachRecord]
    quality: list[QualityEvent]
    views: dict[str, BookView]  # request_id -> the one view every strategy saw
    counters: dict[str, Counter]  # per symbol


# --- persistent results (design §11) ----------------------------------------------------------------

SCHEMA_VERSION = 1
MANIFEST = "manifest.json"
DATA_FILES = ("schedule.jsonl", "orders.jsonl", "fills.jsonl", "quality.jsonl")
VOLATILE_MANIFEST_KEYS = ("runtime",)  # everything else in the manifest is reproducible


def jsonable(v, strict: bool = True):
    """Plain JSON values: Decimal -> fixed-point string (never a float), Enum -> value, Path -> str.
    `strict` rejects floats, so a price or quantity can never lose precision on its way to a row."""
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, Enum):
        return v.value
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, dict):
        return {str(jsonable(k, strict)): jsonable(x, strict) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [jsonable(x, strict) for x in (sorted(v) if isinstance(v, (set, frozenset)) else v)]
    if isinstance(v, float) and strict:
        raise TypeError(f"float {v!r} in a result row: prices and quantities must stay Decimal")
    return v


def dumps_row(row: dict) -> str:
    """One JSONL line; key order is the row's own (fixed by the writer), so output is stable."""
    return json.dumps(jsonable(row), separators=(",", ":"), ensure_ascii=False)


def stable_hash(obj) -> str:
    """sha256 of the canonical JSON form (sorted keys), for configuration and run identity."""
    text = json.dumps(jsonable(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def file_info(path: Path) -> dict:
    """Identity of an input recording: name, size and content hash (the path is informational)."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return {"name": path.name, "path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def git_state(repo: Path) -> dict:
    """Commit and working-tree state of the code that produced a run. `dirty` True means the
    results cannot be attributed to `commit` alone; None means git could not be asked."""
    def git(*args):
        try:
            out = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    commit, branch, status = git("rev-parse", "HEAD"), git("rev-parse", "--abbrev-ref", "HEAD"), git(
        "status", "--porcelain")
    changed = None if status is None else sorted(line[3:] for line in status.splitlines() if line.strip())
    return {
        "commit": commit.strip() if commit else None,
        "branch": branch.strip() if branch else None,
        "dirty": None if changed is None else bool(changed),
        "changed_files": changed,
    }


def config_dict(cfg) -> dict:
    """SimConfig as plain values, in field order."""
    return {name: getattr(cfg, name) for name in cfg.__dataclass_fields__}


def quality_summary(result: "ReplayResult") -> dict:
    """Counts for the manifest: quality records by kind, and distinct incidents per symbol."""
    kinds: dict[str, Counter] = {}
    incidents: dict[str, set] = {}
    for q in result.quality:
        kinds.setdefault(q.symbol, Counter())[q.kind] += 1
        if q.incident is not None:
            incidents.setdefault(q.symbol, set()).add(q.incident)
    return {
        "events_by_kind": {sym: dict(sorted(c.items())) for sym, c in sorted(kinds.items())},
        "incidents": {sym: len(v) for sym, v in sorted(incidents.items())},
        "counters": {sym: dict(sorted(c.items())) for sym, c in sorted(result.counters.items())},
    }


def report_summary(report) -> dict:
    """The headline numbers of a metrics.EvaluationReport as plain values (rates as [num, den]).
    Nothing is recomputed here: every number is read from the report."""
    def rate(r):
        return [r.num, r.den]

    out = {"requests": report.request_counts, "strategies": {}}
    for name in report.strategies:
        lc = report.lifecycle[name]
        row = {
            "eligible": lc.eligible, "strategy_skips": lc.strategy_skips,
            "strategy_skip_reasons": lc.strategy_skip_reasons, "orders_created": lc.orders_created,
            "insertion_skips": lc.insertion_skips, "insertion_skip_reasons": lc.insertion_skip_reasons,
            "placed": lc.placed, "expired": lc.expired, "censored": lc.censored,
            "censored_reasons": lc.censored_reasons, "crossed": lc.crossed,
            "inconsistent_crossings": lc.inconsistent_crossings, "policies": {},
        }
        for p in report.policies:
            s = report.summary[(name, p)]
            row["policies"][p] = {
                "filled_any": s.filled_any, "filled_full": s.filled_full, "partial": s.partial,
                "unfilled_observed": s.unfilled_observed, "censored_unfilled": s.censored_unfilled,
                "resolved_by_cross": s.resolved_by_cross, "crossed_unfilled": s.crossed_unfilled,
                "fill_given_placed": rate(s.fill_given_placed),
                "fill_given_placed_lower": rate(s.fill_given_placed_lower),
                "fill_given_placed_upper": rate(s.fill_given_placed_upper),
                "fill_per_opportunity": rate(s.fill_per_opportunity),
                "fill_common_support": rate(s.fill_common_support),
                "n_fills": s.n_fills, "fills_after_cross": s.fills_after_cross,
                "legacy_filled_any": s.legacy_filled_any,
                "legacy_fills_after_cross": s.legacy_fills_after_cross,
                "legacy_first_fill_after_cross": s.legacy_first_fill_after_cross,
                "fill_qty_by_evidence": s.fill_qty_by_evidence,
                "markout_mean_bps": {str(d): m.mean for d, m in s.markouts.items()},
                "markout_n": {str(d): m.n_valid for d, m in s.markouts.items()},
            }
        out["strategies"][name] = row
    return out


class ResultWriter:
    """Writes one run directory. Call write_schedule() before the replay, write() after the
    analysis, then write_manifest(); each returns what the manifest should record about it."""

    def __init__(self, run_dir: Path, run_id: str, overwrite: bool = False):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.files: dict[str, dict] = {}  # name -> {rows, sha256}
        existing = [p.name for p in self.run_dir.iterdir()] if self.run_dir.is_dir() else []
        if existing and not overwrite:
            raise FileExistsError(f"{self.run_dir} already holds a run ({', '.join(sorted(existing))}); "
                                  "choose another run id or pass overwrite")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for name in (*DATA_FILES, MANIFEST):  # only this writer's own files are ever replaced
            (self.run_dir / name).unlink(missing_ok=True)

    # --- files ------------------------------------------------------------------------------

    def write_schedule(self, schedule: Iterable) -> dict:
        return self._write("schedule.jsonl", (
            {"schema_version": SCHEMA_VERSION, "request_id": r.request_id, "symbol": r.symbol, "side": r.side,
             "ts_ns": r.ts_ns} for r in schedule))

    def write(self, result: "ReplayResult", report, cfg) -> dict[str, dict]:
        """orders.jsonl, fills.jsonl and quality.jsonl from a replay and its EvaluationReport.

        The report supplies the per-policy settlement (under its evidence mode and crossing rule)
        and the request eligibility; the replay supplies the orders, candidates and quality events.
        """
        records = {(rec.request.request_id, rec.strategy): rec for rec in [*result.orders, *result.skips]}
        status = {r.request.request_id: r for r in report.requests}
        reaches = {(r.order_id, r.reach.ref): r for r in result.reaches}
        self._write("orders.jsonl", self._order_rows(report, records, status, result, cfg))
        self._write("fills.jsonl", self._fill_rows(report, records, reaches))
        self._write("quality.jsonl", self._quality_rows(result))
        return self.files

    def write_manifest(self, manifest: dict) -> Path:
        """`manifest` must keep everything that varies between identical runs under VOLATILE_MANIFEST_KEYS."""
        doc = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, **manifest, "files": self.files}
        path = self.run_dir / MANIFEST
        volatile = {k: doc.pop(k) for k in VOLATILE_MANIFEST_KEYS if k in doc}
        doc = {**jsonable(doc), **jsonable(volatile, strict=False)}  # floats only in the volatile block
        path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    def _write(self, name: str, rows: Iterable[dict]) -> dict:
        digest, n = hashlib.sha256(), 0
        with open(self.run_dir / name, "w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                line = dumps_row(row) + "\n"
                fh.write(line)
                digest.update(line.encode())
                n += 1
        self.files[name] = {"rows": n, "sha256": digest.hexdigest()}
        return self.files[name]

    # --- rows -------------------------------------------------------------------------------

    def _fill_model(self, report, cfg) -> str:
        mode = "l3_trades" if report.evidence_mode == "trades" else "l3_only"
        return f"{mode},cross_rule={report.cross_rule},increase_policy={cfg.increase_policy}"

    def _order_rows(self, report, records, status, result, cfg) -> Iterator[dict]:
        trades = report.evidence_mode == "trades"
        fill_model = self._fill_model(report, cfg)
        H = report.horizon_ns
        for obs in report.orders:  # requests in time order, strategies in run order
            req = obs.request
            rec = records[(req.request_id, obs.strategy)]
            o = rec if isinstance(rec, SimulatedOrder) else None
            placed = o is not None and o.inserted_ts is not None
            r = status[req.request_id]
            view = result.views.get(req.request_id)
            row = {
                "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                "strategy": obs.strategy, "symbol": req.symbol, "request_id": req.request_id,
                "order_id": f"{req.request_id}/{obs.strategy}", "side": req.side,
                "decision_ts_ns": req.ts_ns, "placement_ts_ns": o.inserted_ts if placed else None,
                "placed": placed,
                "skip_reason": None if placed else (o.skip_reason if o is not None else rec.reason),
                "skip_stage": None if placed else ("insertion" if o is not None else "decision"),
                "request_status": r.status, "request_status_reason": r.reason, "eligible": obs.eligible,
                "best_bid_at_decision": view.best_bid if view else None,
                "best_ask_at_decision": view.best_ask if view else None,
                "tick_size": view.tick if view else None,
                "limit_price": o.price if o is not None else None,
                "order_qty": o.qty if o is not None else cfg.order_qty[req.symbol],
                "queue_ahead_qty": o.queue_ahead_qty0 if placed else None,
                "queue_ahead_orders": o.queue_ahead_orders0 if placed else None,
                "touch_qty": o.touch_qty0 if placed else None,
                "level_rank": o.level_rank0 if placed else None,
                "max_level_rank": o.max_level_rank if placed else None,
                "edge_bps": obs.edge_bps,
                "horizon_ns": H, "end_ts_ns": o.end_ts if placed else None,
                "end_reason": o.end_reason if placed else None,
                "censored": obs.censored if placed else None,
                "checksum_covered": o.checksum_covered if placed else None,
            }
            for p in report.policies:
                out = obs.outcomes[p] if placed else None
                first, done = (out.first_fill_ts, out.complete_ts) if out else (None, None)
                row[f"filled_qty_{p}"] = out.filled_qty if out else None
                row[f"remaining_qty_{p}"] = o.qty - out.filled_qty if out else None
                row[f"status_{p}"] = out.status if out else None
                row[f"first_fill_ts_ns_{p}"] = first
                row[f"complete_ts_ns_{p}"] = done
                row[f"time_to_first_fill_ns_{p}"] = first - o.inserted_ts if first is not None else None
                row[f"time_to_complete_fill_ns_{p}"] = done - o.inserted_ts if done is not None else None
                row[f"resolved_by_cross_{p}"] = out.resolved_by_cross if out else None
            window = [c for c in o.fills if c.ts_ns < o.inserted_ts + H] if placed else []
            ranks = [int(part.evidence) for c in window
                     for part in (c.attribution if trades else (c,))]  # a candidate has .evidence too
            cross = o.cross if placed else None
            row.update({
                "n_candidates": len(o.fills) if placed else None,
                "n_candidates_in_horizon": len(window) if placed else None,
                "max_evidence_rank": max(ranks) if ranks else None,
                "crossed": cross is not None if placed else None,
                "crossed_in_horizon": obs.cross_ts is not None if placed else None,
                "cross_ts_ns": cross.ts_ns if cross else None,
                "cross_event_ref": cross.ref if cross else None,
                "cross_event_kind": cross.kind if cross else None,
                "cross_price": cross.price if cross else None,
                "cross_qty": cross.qty if cross else None,
                "cross_ahead_qty": cross.ahead_qty if cross else None,
                "n_crossings": o.diag.get("crossing", 0) if placed else None,
                "inconsistent_crossing": o.inconsistent_crossing if placed else None,
                "diag": dict(sorted(o.diag.items())) if placed else {},
                "fill_model": fill_model,
            })
            yield row

    def _fill_rows(self, report, records, reaches) -> Iterator[dict]:
        H = report.horizon_ns
        for obs in report.orders:
            if obs.outcomes is None:
                continue  # never placed: no candidates
            o = records[(obs.request.request_id, obs.strategy)]
            accepted = {p: {sf.candidate.event_ref: sf.qty for sf in obs.outcomes[p].fills} for p in report.policies}
            cross = o.cross
            for c in o.fills:  # stream order
                reach = reaches.get((o.order_id, c.event_ref))
                row = {
                    "schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                    "order_id": o.order_id, "strategy": o.strategy, "symbol": obs.request.symbol, "side": o.side,
                    "fill_ts_ns": c.ts_ns, "since_placement_ns": c.ts_ns - o.inserted_ts,
                    "fill_price": c.price, "fill_qty": c.qty,
                    "event_price": reach.reach.event_price if reach else None,
                    "mechanism": c.mechanism, "evidence": c.evidence.name, "evidence_rank": int(c.evidence),
                    "event_ref": c.event_ref,
                    "source_file": reach.source_file if reach else None,
                    "source_line": reach.source_line if reach else None,
                    "trade_ids": c.trade_ids,
                    "trade_evidence": None if c.attribution is None else [
                        {"qty": part.qty, "evidence": part.evidence.name, "evidence_rank": int(part.evidence),
                         "trade_ids": part.trade_ids} for part in c.attribution],
                    "ahead_qty_at_fill": reach.ahead_qty if reach else None,
                    "checksum_covered": reach.checksum_covered if reach else None,
                    "in_horizon": c.ts_ns < o.inserted_ts + H,
                    "is_cross_event": cross is not None and c.event_ref == cross.ref,
                    "after_cross": cross is not None and c.ts_ns > cross.ts_ns,
                }
                for p in report.policies:
                    row[f"accepted_qty_{p}"] = accepted[p].get(c.event_ref, Decimal(0))
                yield row

    def _quality_rows(self, result) -> Iterator[dict]:
        for q in result.quality:  # stream order
            detail = {k: (Path(v).name if k == "file" else v) for k, v in q.detail.items()}
            yield {"schema_version": SCHEMA_VERSION, "run_id": self.run_id, "ts_ns": q.ts_ns, "symbol": q.symbol,
                   "kind": q.kind, "incident": q.incident, "seq": q.seq, "file": q.file, "line_no": q.line_no,
                   "recv_ns": q.recv_ns, "detail": detail}
