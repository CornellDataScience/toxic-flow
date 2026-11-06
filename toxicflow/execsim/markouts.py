"""Mid-price sampling and signed markouts (design §11.5).

A markout needs the mid at t + D, after the fill. sample_mids() reads it by
replaying the recording a second time through HistoricalReplayEngine with a
probe strategy that never places, so a mid at time t obeys exactly the rules a
decision at t does: the book holds only items with timestamp < t, and the
engine's data gate refuses a time that is unsynced (before the first snapshot,
after a checksum failure or session restart), inside a receive gap, or after
the end of the data. A refused time has no mid; it is never forward-filled
from an earlier one. The probe run is separate from the strategy run, so future
prices can never reach a decision.

Signed markout in basis points, positive = favorable to us:

    markout_bps = sign * (mid(t_fill + D) - fill_price) / fill_price * 10_000,  sign = +1 buy, -1 sell

The denominator is the fill (limit) price, as in the §11.5 formula. The starter
analysis/markouts.py on the implementation branch divides by the mid at fill;
the design leaves the choice to the team (§13.3). The two differ only in the
second order (about half the spread in relative terms).
"""

import dataclasses
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from .config import SimConfig
from .models import BookView, PlacementRequest, Side, Skip, SkipReason
from .reader import StreamRecord
from .replay import HistoricalReplayEngine
from .timeutil import MS, S

BPS = Decimal(10_000)
DEFAULT_MARKOUT_HORIZONS_NS = (100 * MS, 1 * S, 5 * S, 10 * S, 60 * S)  # §11.5 set plus 5 s


@dataclass(frozen=True, slots=True)
class MidSample:
    symbol: str
    ts_ns: int
    mid: Decimal | None
    missing: str | None  # why mid is None: an engine SkipReason (UNSYNCED, DATA_GAP, NO_DATA), ONE_SIDED or CROSSED


class _MidProbe:
    """Never places: the engine records the shared BookView for every request that passes its data gate."""

    name = "mid_probe"

    def decide(self, req: PlacementRequest, view: BookView) -> Skip:
        return Skip(SkipReason.NO_TOUCH)


def view_mid(view: BookView) -> tuple[Decimal | None, str | None]:
    """Mid of a view, or None with a reason when either side is empty or the book is locked or crossed."""
    bid, ask = view.best_bid, view.best_ask
    if bid is None or ask is None:
        return None, "ONE_SIDED"
    if bid >= ask:
        return None, "CROSSED"
    return (bid + ask) / 2, None


def sample_mids(cfg: SimConfig, times: Iterable[tuple[str, int]], files: Sequence[Path] | None = None,
                records: Iterable[StreamRecord] | None = None) -> dict[tuple[str, int], MidSample]:
    """Mid at each (symbol, ts_ns), from `files` or an already-built record stream.

    Uses `cfg`'s symbols, gap_ns and tick settings; warmup and latency do not apply
    to a price observation.
    """
    if (files is None) == (records is None):
        raise ValueError("give exactly one of files or records")
    wanted = sorted({(sym, int(ts)) for sym, ts in times})
    probe_cfg = dataclasses.replace(cfg, latency_ns=0, warmup_ns=0, view_depth=1, audit_invariants=False)
    reqs = [PlacementRequest(f"mid|{sym}|{ts}", sym, Side.BUY, ts) for sym, ts in wanted]
    eng = HistoricalReplayEngine(probe_cfg, [_MidProbe()], reqs)
    result = eng.run(files) if files is not None else eng.replay(records)
    gate = {s.request.request_id: s.reason.value for s in result.skips if s.reason is not SkipReason.NO_TOUCH}
    out = {}
    for r in reqs:
        view = result.views.get(r.request_id)
        mid, missing = view_mid(view) if view is not None else (None, gate.get(r.request_id, "NO_VIEW"))
        out[(r.symbol, r.ts_ns)] = MidSample(r.symbol, r.ts_ns, mid, missing)
    return out


def signed_markout_bps(side: Side, price: Decimal, future_mid: Decimal) -> Decimal:
    """sign * (future_mid - price) / price * 10_000; positive when the market moved in our favor."""
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}")
    return side.sign * (future_mid - price) / price * BPS


def edge_bps(side: Side, price: Decimal, mid: Decimal) -> Decimal:
    """Price improvement versus the decision mid (§11.5): sign * (mid - price) / mid * 10_000."""
    return side.sign * (mid - price) / mid * BPS


def drift_bps(side: Side, mid_now: Decimal, future_mid: Decimal) -> Decimal:
    """Signed mid move with no fill: sign * (mid(t + D) - mid(t)) / mid(t) * 10_000 (adverse-selection baseline)."""
    return side.sign * (future_mid - mid_now) / mid_now * BPS
