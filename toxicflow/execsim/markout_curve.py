"""Markout curves: how much of the spread a passive fill keeps, and for how long.

Every passive fill earns the spread and pays adverse selection (and a fee, not modeled yet):

    net(d) = spread + adverse(d) [- fee]                        (bps, positive = good for us)

    spread     = sign * (mid0 - price) / mid0 * 1e4             what the fill earned on arrival (>= 0)
    adverse(d) = sign * (mid(t + d) - mid0) / mid0 * 1e4        how far the mid moved against us by t + d
    sign = +1 for a buy, -1 for a sell

mid0 is the mid just before the fill *with our order in the book* (mid_with_order): the quote is
sampled at the fill time, whose book holds only items stamped before it, then our resting price
is merged in as the touch. Without that step a fill after the real queue at our price emptied
looks like it paid the spread. All three terms share the mid0 denominator, so the decomposition
is exact (metrics.signed_markout_bps divides by the price instead; the two differ
by about spread * spread, under 0.001 bps here).

The curve is the mean adverse(d) on a log grid of d (10 ms to 10 s by default). With toxic flow
it slopes down: after we buy, the price falls. The horizontal line at -mean(spread) is the
cushion the spread gives us; the breakeven horizon is where the curve crosses it, i.e. where the
mean net markout reaches zero and the average fill starts losing money.

Two fill models are compared on the same hypothetical orders (same schedule, strategy, size,
horizon):

    queue_aware  execsim's fills: the order joins the back of the real queue and fills only from
                 volume that reaches it after everything ahead is gone (a settlement policy:
                 conservative <= balanced <= optimistic)
    naive        the whole order fills the first time the price touches it: a trade prints at or
                 through our price on our side of the book, or an opposite order comes to rest
                 at or through it (the crossing). Queue position is ignored, which is the usual
                 optimistic shortcut this project is meant to replace.

Means are quantity-weighted (PnL per unit traded), so a 6e-5 partial does not count as much as
a full fill. Distributions and concentration are per fill. Standard errors are clustered by
fill time: overlapping orders filled by one market event are one observation, not many.
The curve uses only fills with a mid at every grid point, so its composition is the same at
every horizon; dropped fills are counted.
"""

import bisect
import dataclasses
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from .config import SimConfig
from .metrics import PLACED, EvaluationReport
from .models import BookView, PlacementRequest, Side, Skip, SkipReason
from .replay import HistoricalReplayEngine
from .timeutil import MS, S
from .trades import TradeIndex

BPS = Decimal(10_000)
QUEUE_AWARE, NAIVE = "queue_aware", "naive"


def log_horizons(lo_ns: int = 10 * MS, hi_ns: int = 10 * S, per_decade: int = 6) -> tuple[int, ...]:
    """Log-spaced horizons from lo to hi inclusive: 10 ms, 14.7 ms, 21.5 ms, ... 10 s by default."""
    if not 0 < lo_ns < hi_ns:
        raise ValueError(f"need 0 < lo < hi, got {lo_ns}, {hi_ns}")
    n = round(math.log10(hi_ns / lo_ns) * per_decade)
    return tuple(sorted({int(round(lo_ns * (hi_ns / lo_ns) ** (k / n))) for k in range(n + 1)}))


def horizon_label(ns: int) -> str:
    if ns >= S:
        return f"{ns / S:g}s"
    if ns >= MS:
        return f"{ns / MS:.3g}ms"
    return f"{ns / 1000:g}us"


# --- fills -------------------------------------------------------------------------------------------------


FILL_COLUMNS = ["model", "fill_id", "order_id", "symbol", "side", "sign", "ts_ns", "price", "qty",
                "since_insert_ns", "queue_ahead_qty0", "queue_ahead_orders0", "trigger", "event_qty", "through"]


def queue_aware_fills(report: EvaluationReport, policy: str, strategy: str) -> pd.DataFrame:
    """Settled fills of `strategy` under `policy` (eligible placed orders only, as analyze() keeps them)."""
    orders = {o.order_id: o for o in report.orders if o.strategy == strategy}
    rows = []
    for k, f in enumerate(x for x in report.fills if x.policy == policy and x.strategy == strategy):
        o = orders[f.order_id]
        rows.append((f"{QUEUE_AWARE}:{policy}", f"{f.order_id}#{k}", f.order_id, f.symbol, f.side.value, f.side.sign,
                     f.ts_ns, f.price, f.qty, f.since_insert_ns, o.queue_ahead_qty0, o.queue_ahead_orders0,
                     f.mechanism.value, f.candidate_qty, False))
    return pd.DataFrame(rows, columns=FILL_COLUMNS)


@dataclass(frozen=True)
class NaiveTouch:
    ts_ns: int
    trigger: str  # "trade" or "cross"
    event_qty: Decimal  # the trade's size, or the crossing order's
    through: bool  # the trade or quote went strictly through our price


def naive_touch(side_sign: int, price: Decimal, book_side: str, start_ns: int, stop_ns: int,
                trades: Sequence, trade_ts: Sequence[int], cross_ts: int | None = None,
                cross_price: Decimal | None = None, cross_qty: Decimal | None = None) -> NaiveTouch | None:
    """First time in [start_ns, stop_ns) the price touches a resting order at `price`.

    trades/trade_ts: one symbol's trades.Trade list sorted by time, and their times. A trade
    touches us when it executed against our book side at or through our price (a buy at P is
    touched by a sell-aggressor print at <= P). cross_ts: the order's first crossing, if any.
    """
    best = None
    for i in range(bisect.bisect_left(trade_ts, start_ns), len(trades)):
        t = trades[i]
        if t.ts_ns >= stop_ns:
            break
        if t.maker_book_side == book_side and side_sign * (t.price - price) <= 0:
            best = NaiveTouch(t.ts_ns, "trade", t.qty, t.price != price)
            break
    if cross_ts is not None and start_ns <= cross_ts < stop_ns and (best is None or cross_ts < best.ts_ns):
        best = NaiveTouch(cross_ts, "cross", cross_qty if cross_qty is not None else Decimal(0),
                          cross_price is not None and cross_price != price)
    return best


def naive_fills(report: EvaluationReport, trades: TradeIndex, strategy: str,
                crossings: dict[str, tuple[Decimal, Decimal]] | None = None) -> tuple[pd.DataFrame, dict]:
    """Naive fills for the same eligible placed orders the queue-aware model saw.

    The window is [inserted, inserted + H), cut at the censoring time for an order whose data
    broke before H (its later outcome is unknown, so a later touch is not counted).
    crossings: order_id -> (price, qty) of the crossing order, for the trigger columns.
    """
    H = report.horizon_ns
    times = {sym: [t.ts_ns for t in ts] for sym, ts in trades.trades.items()}
    rows, counts = [], {"orders": 0, "filled": 0, "by_trade": 0, "by_cross": 0}
    for o in report.orders:
        if o.strategy != strategy or not o.eligible or o.stage != PLACED:
            continue
        counts["orders"] += 1
        stop = o.end_ts if o.censored else o.inserted_ts + H
        cp, cq = (crossings or {}).get(o.order_id, (None, None))
        sym_trades = trades.trades.get(o.request.symbol, [])
        touch = naive_touch(o.request.side.sign, o.price, o.request.side.book_side, o.inserted_ts, stop,
                            sym_trades, times.get(o.request.symbol, []), o.cross_ts, cp, cq)
        if touch is None:
            continue
        counts["filled"] += 1
        counts[f"by_{touch.trigger}"] += 1
        rows.append((NAIVE, f"{o.order_id}#0", o.order_id, o.request.symbol, o.request.side.value,
                     o.request.side.sign, touch.ts_ns, o.price, o.qty, touch.ts_ns - o.inserted_ts,
                     o.queue_ahead_qty0, o.queue_ahead_orders0, touch.trigger, touch.event_qty, touch.through))
    return pd.DataFrame(rows, columns=FILL_COLUMNS), counts


def fill_rates(report: EvaluationReport, strategy: str, policy: str, naive: pd.DataFrame,
               naive_counts: dict) -> pd.DataFrame:
    """Orders filled at all, per model, over the same eligible placed orders.

    Queue-aware orders whose outcome is unknown (censored or stopped at a crossing, unfilled)
    are left out of its denominator, as metrics.FillSummary.fill_given_placed does.
    """
    placed = [o for o in report.orders if o.strategy == strategy and o.eligible and o.stage == PLACED]
    filled = [o for o in placed if o.outcomes[policy].filled_qty > 0]
    unknown = [o for o in placed if o.outcomes[policy].filled_qty == 0
               and (o.censored or o.outcomes[policy].resolved_by_cross)]
    qa_t = [o.outcomes[policy].first_fill_ts - o.inserted_ts for o in filled]
    known = len(placed) - len(unknown)
    rows = [
        {"model": f"{QUEUE_AWARE}:{policy}", "orders": len(placed), "unknown_outcome": len(unknown),
         "filled_orders": len(filled), "fill_rate": len(filled) / known if known else np.nan,
         "median_time_to_fill_ms": float(np.median(qa_t)) / MS if qa_t else np.nan},
        {"model": NAIVE, "orders": naive_counts["orders"], "unknown_outcome": 0,
         "filled_orders": naive_counts["filled"],
         "fill_rate": naive_counts["filled"] / naive_counts["orders"] if naive_counts["orders"] else np.nan,
         "median_time_to_fill_ms": float(naive["since_insert_ns"].median()) / MS if len(naive) else np.nan},
    ]
    return pd.DataFrame(rows)


# --- mids and the decomposition ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Quote:
    """Best bid / ask at a time, from the same as-of view a decision sees. None when unavailable."""

    bid: Decimal | None
    ask: Decimal | None
    missing: str | None  # engine SkipReason when the data gate refused the time, or ONE_SIDED / CROSSED

    @property
    def mid(self) -> Decimal | None:
        return None if self.missing else (self.bid + self.ask) / 2


class _QuoteProbe:
    """Never places; the engine records the shared BookView for every time its data gate accepts."""

    name = "quote_probe"

    def decide(self, req: PlacementRequest, view: BookView) -> Skip:
        return Skip(SkipReason.NO_TOUCH)


def sample_quotes(cfg: SimConfig, times: Iterable[tuple[str, int]], files: Sequence[Path]) -> dict[tuple[str, int], Quote]:
    """Best bid / ask at each (symbol, ts), with markouts.sample_mids' rules: the book holds only
    items stamped before ts, and a time the data gate refuses (unsynced, gap, past the data) has
    no quote rather than a stale one."""
    wanted = sorted({(sym, int(ts)) for sym, ts in times})
    probe_cfg = dataclasses.replace(cfg, latency_ns=0, warmup_ns=0, view_depth=1, audit_invariants=False)
    reqs = [PlacementRequest(f"quote|{sym}|{ts}", sym, Side.BUY, ts) for sym, ts in wanted]
    result = HistoricalReplayEngine(probe_cfg, [_QuoteProbe()], reqs).run(files)
    gate = {k.request.request_id: k.reason.value for k in result.skips if k.reason is not SkipReason.NO_TOUCH}
    out = {}
    for r in reqs:
        view = result.views.get(r.request_id)
        if view is None:
            out[(r.symbol, r.ts_ns)] = Quote(None, None, gate.get(r.request_id, "NO_VIEW"))
            continue
        bid, ask = view.best_bid, view.best_ask
        missing = "ONE_SIDED" if bid is None or ask is None else "CROSSED" if bid >= ask else None
        out[(r.symbol, r.ts_ns)] = Quote(bid, ask, missing)
    return out


def required_times(fills: pd.DataFrame, horizons: Iterable[int]) -> set[tuple[str, int]]:
    """(symbol, ts) pairs to sample: each fill time (the quote at execution) and fill time + d."""
    ds = (0, *horizons)
    return {(sym, int(ts) + d) for sym, ts in zip(fills["symbol"], fills["ts_ns"]) for d in ds}


def mid_with_order(q: Quote, sign: int, price: Decimal) -> tuple[Decimal | None, str | None]:
    """Mid of the book with our resting order added: a buy at P makes the bid max(bid, P), a
    sell makes the ask min(ask, P).

    The recording never contained our order. When the real queue at our price empties, the
    recorded touch moves past us and the recorded mid can sit on the wrong side of our price,
    which would show a passive fill as paying the spread. With our order in the book it is the
    touch, so the spread earned is never negative.
    """
    if q.bid is None or q.ask is None:
        return None, q.missing or "ONE_SIDED"
    bid, ask = (max(q.bid, price), q.ask) if sign > 0 else (q.bid, min(q.ask, price))
    if bid >= ask:
        return None, "CROSSED_WITH_ORDER"
    return (bid + ask) / 2, None


def decompose(fills: pd.DataFrame, quotes: dict[tuple[str, int], Quote], horizons: Sequence[int]) -> pd.DataFrame:
    """One row per (fill, horizon): spread, adverse and net in bps and in quote currency.

    mid0 is the mid at execution with our order in the book (mid_with_order); mid(t + d) is the
    recorded mid. *_bps are per fill; *_usd per unit of base asset; pnl_usd = net_usd * qty.
    Rows with a missing quote carry NaN and the reason in `missing`.
    """
    out = []
    for f in fills.itertuples(index=False):
        q0 = quotes.get((f.symbol, int(f.ts_ns)))
        m0, why0 = mid_with_order(q0, f.sign, f.price) if q0 is not None else (None, "NOT_SAMPLED")
        for d in horizons:
            qd = quotes.get((f.symbol, int(f.ts_ns) + d))
            md = qd.mid if qd is not None else None
            row = {"model": f.model, "fill_id": f.fill_id, "horizon_ns": d}
            if m0 is None or md is None:
                why = why0 or ("NOT_SAMPLED" if qd is None else qd.missing)
                row.update(spread_bps=np.nan, adverse_bps=np.nan, net_bps=np.nan, spread_usd=np.nan,
                           adverse_usd=np.nan, net_usd=np.nan, pnl_usd=np.nan, missing=why)
            else:
                s = f.sign * (m0 - f.price)  # exact Decimal differences, converted once
                a = f.sign * (md - m0)
                row.update(spread_bps=float(s / m0 * BPS), adverse_bps=float(a / m0 * BPS),
                           net_bps=float((s + a) / m0 * BPS), spread_usd=float(s), adverse_usd=float(a),
                           net_usd=float(s + a), pnl_usd=float((s + a) * f.qty), missing=None)
            out.append(row)
    long = pd.DataFrame(out, columns=["model", "fill_id", "horizon_ns", "spread_bps", "adverse_bps", "net_bps",
                                      "spread_usd", "adverse_usd", "net_usd", "pnl_usd", "missing"])
    keep = ["model", "fill_id", "order_id", "symbol", "side", "ts_ns", "price", "qty", "since_insert_ns",
            "queue_ahead_qty0", "queue_ahead_orders0", "trigger", "event_qty", "through"]
    long = long.merge(fills[keep], on=["model", "fill_id"], how="left")
    long["qty"] = long["qty"].astype(float)
    return long


def complete_panel(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Keep fills with a markout at every horizon. Returns (panel, dropped fills per model)."""
    ok = long.groupby(["model", "fill_id"])["net_bps"].transform(lambda s: s.notna().all())
    dropped = long[~ok].groupby("model")["fill_id"].nunique()
    return long[ok].copy(), dropped


# --- curve, breakeven, PnL split -----------------------------------------------------------------------------


def _wmean_se(x: np.ndarray, w: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Weighted mean and its cluster-robust standard error (clusters = fill times)."""
    sw = w.sum()
    if sw <= 0:
        return np.nan, np.nan
    m = float((w * x).sum() / sw)
    resid = pd.Series(w * (x - m)).groupby(cluster).sum().to_numpy()
    g = len(resid)
    se = float(np.sqrt((resid ** 2).sum() * g / (g - 1)) / sw) if g > 1 else np.nan
    return m, se


def curve_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (model, horizon): qty-weighted mean spread / adverse / net with clustered SEs, and
    per-fill percentiles and share of losing fills."""
    rows = []
    for (model, d), g in panel.groupby(["model", "horizon_ns"], sort=True):
        w, cl = g["qty"].to_numpy(), g["ts_ns"].to_numpy()
        row = {"model": model, "horizon_ns": d, "horizon": horizon_label(d), "n_fills": len(g),
               "n_events": g["ts_ns"].nunique(), "qty": w.sum()}
        for k in ("spread", "adverse", "net"):
            row[f"{k}_bps"], row[f"{k}_se"] = _wmean_se(g[f"{k}_bps"].to_numpy(), w, cl)
        row["net_usd_per_unit"] = float((g["net_usd"] * w).sum() / w.sum())
        row["pnl_usd"] = float(g["pnl_usd"].sum())
        for q in (5, 25, 50, 75, 95):
            row[f"net_p{q}"] = float(np.percentile(g["net_bps"], q))
        row["frac_losing"] = float((g["net_bps"] < 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Breakeven:
    model: str
    horizon_ns: float | None  # interpolated in log time; None if the mean net never reaches 0 on the grid
    status: str  # "crossed", "before_grid" (already <= 0 at the first point), "never" (> 0 throughout)

    @property
    def label(self) -> str:
        if self.status == "crossed":
            return horizon_label(int(self.horizon_ns))
        return {"before_grid": "before the first horizon", "never": "not within the grid"}[self.status]


def breakeven(curve: pd.DataFrame) -> dict[str, Breakeven]:
    """Where mean net markout first reaches zero (the curve meets the -half-spread line)."""
    out = {}
    for model, g in curve.sort_values("horizon_ns").groupby("model"):
        d, net = g["horizon_ns"].to_numpy(float), g["net_bps"].to_numpy(float)
        if len(net) == 0 or np.isnan(net).all():
            out[model] = Breakeven(model, None, "never")
            continue
        if net[0] <= 0:
            out[model] = Breakeven(model, None, "before_grid")
            continue
        hit = np.nonzero(net <= 0)[0]
        if not len(hit):
            out[model] = Breakeven(model, None, "never")
            continue
        k = hit[0]
        x0, x1 = np.log(d[k - 1]), np.log(d[k])
        frac = net[k - 1] / (net[k - 1] - net[k])
        out[model] = Breakeven(model, float(np.exp(x0 + frac * (x1 - x0))), "crossed")
    return out


def pnl_split(curve: pd.DataFrame, horizons: Sequence[int]) -> pd.DataFrame:
    """The PnL decomposition at selected horizons: spread + adverse selection + fee = net (bps,
    qty-weighted), the share of the spread that survives, and total PnL in quote currency."""
    t = curve[curve["horizon_ns"].isin(horizons)].copy()
    t["fee_bps"] = 0.0  # maker fee/rebate: not modeled yet
    t["spread_kept"] = np.where(t["spread_bps"] > 0, t["net_bps"] / t["spread_bps"], np.nan)
    return t[["model", "horizon", "n_fills", "spread_bps", "adverse_bps", "fee_bps", "net_bps", "net_se",
              "spread_kept", "frac_losing", "pnl_usd"]].reset_index(drop=True)


# --- concentration and the worst fills -----------------------------------------------------------------------


def concentration(panel: pd.DataFrame, horizon_ns: int, shares: Sequence[float] = (0.01, 0.05, 0.10, 0.20, 0.50)
                  ) -> pd.DataFrame:
    """How much of the total loss the worst fills account for, at one horizon (by pnl_usd).

    loss_share: the worst k of fills' losses / all losing fills' losses.
    net_share: the worst k of fills' PnL / total net PnL (a value > 1 means the rest made money).
    """
    rows = []
    for model, g in panel[panel["horizon_ns"] == horizon_ns].groupby("model"):
        pnl = np.sort(g["pnl_usd"].to_numpy())  # worst first
        losses = -pnl[pnl < 0]
        total_loss, total = losses.sum(), pnl.sum()
        row = {"model": model, "horizon": horizon_label(horizon_ns), "n_fills": len(pnl),
               "frac_losing": float((pnl < 0).mean()) if len(pnl) else np.nan,
               "total_loss_usd": float(total_loss), "net_pnl_usd": float(total)}
        for s in shares:
            k = max(1, int(math.ceil(s * len(pnl)))) if len(pnl) else 0
            worst = pnl[:k]
            row[f"worst_{s:.0%}_loss_share"] = float(-worst[worst < 0].sum() / total_loss) if total_loss > 0 else np.nan
            row[f"worst_{s:.0%}_net_share"] = float(worst.sum() / total) if total != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def lorenz(panel: pd.DataFrame, horizon_ns: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(fraction of fills, worst first; cumulative fraction of total loss) per model, for plotting."""
    out = {}
    for model, g in panel[panel["horizon_ns"] == horizon_ns].groupby("model"):
        pnl = np.sort(g["pnl_usd"].to_numpy())
        loss = np.clip(-pnl, 0, None)
        total = loss.sum()
        x = np.arange(1, len(pnl) + 1) / len(pnl) if len(pnl) else np.array([])
        out[model] = (np.r_[0, x], np.r_[0, np.cumsum(loss) / total] if total > 0 else np.r_[0, x * 0])
    return out


PROFILE_FEATURES = ("spread_bps", "since_insert_ms", "queue_ahead_qty0", "queue_ahead_orders0", "event_qty", "qty",
                    "is_buy", "through")


def worst_profile(panel: pd.DataFrame, horizon_ns: int, worst_share: float = 0.10) -> pd.DataFrame:
    """Mean features of the worst `worst_share` of fills (by net bps at the horizon) against the rest.

    A starting point for hypotheses about what makes a fill toxic, not a model: queue position
    at insertion, how long the order waited, the size of the event that filled it, side,
    and (naive) whether the price went through us. Trigger shares are added per trigger value.
    """
    g = panel[panel["horizon_ns"] == horizon_ns].copy()
    g["since_insert_ms"] = g["since_insert_ns"] / MS
    g["is_buy"] = (g["side"] == "buy").astype(float)
    for c in ("queue_ahead_qty0", "event_qty", "qty"):
        g[c] = g[c].astype(float)
    g["through"] = g["through"].astype(float)
    rows = []
    for model, m in g.groupby("model"):
        cut = m["net_bps"].quantile(worst_share)
        groups = {"worst": m[m["net_bps"] <= cut], "rest": m[m["net_bps"] > cut]}
        for name, part in groups.items():
            row = {"model": model, "horizon": horizon_label(horizon_ns), "group": name, "n_fills": len(part),
                   "net_bps": part["net_bps"].mean()}
            row.update({f: part[f].mean() for f in PROFILE_FEATURES})
            for trig, share in part["trigger"].value_counts(normalize=True).items():
                row[f"trigger={trig}"] = share
            rows.append(row)
    return pd.DataFrame(rows)
