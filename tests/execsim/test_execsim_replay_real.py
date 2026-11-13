"""Real-data smoke runs of the replay engine (acceptance checks that need no fill model)."""

from collections import Counter
from decimal import Decimal as D

import pytest

from toxicflow.execsim.config import SimConfig
from toxicflow.execsim.execution import FillRecorder
from toxicflow.execsim.markouts import sample_mids
from toxicflow.execsim.metrics import analyze, format_report, required_mid_times
from toxicflow.execsim.models import Evidence
from toxicflow.execsim.models import Side
from toxicflow.execsim.orders import OrderState
from toxicflow.execsim.reader import KIND_META, RecordingReader, book_items, file_hour, list_files
from toxicflow.execsim.replay import HistoricalReplayEngine
from toxicflow.execsim.results import DISCONTINUITY_KINDS
from toxicflow.execsim.schedule import PlacementSchedule
from toxicflow.execsim.strategies import ImproveQuoteStrategy, JoinBestStrategy, QuoteDeeperStrategy
from toxicflow.execsim.timeutil import S
from toxicflow.execsim.trades import TradeIndex, attribute_trades
from toxicflow.kraken.book import L3Book, PairInfo
from toxicflow.kraken.config import DATA_DIR

L3_FILES = list_files(DATA_DIR / "raw", "l3")


@pytest.mark.skipif(not L3_FILES, reason=f"no L3 recordings under {DATA_DIR / 'raw' / 'l3'}; run scripts/record.py")
def test_engine_on_real_hour():
    path = min(L3_FILES, key=lambda p: p.stat().st_size)
    symbols = ("BTC/USD", "ETH/USD")
    start = int(file_hour(path, "l3").timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, start + 3600 * S, 10 * S)
    cfg = SimConfig(symbols=symbols, order_qty={s: "0.001" for s in symbols}, horizon_ns=10 * S,
                    audit_invariants=True)
    eng = HistoricalReplayEngine(cfg, [JoinBestStrategy()], schedule)
    result = eng.run([path])

    # same counts as validate_recording.py's replay through a plain L3Book
    plain: dict[str, L3Book] = {}
    expected: dict[str, Counter] = {s: Counter() for s in symbols}
    for rec in RecordingReader(files=[path]):
        if rec.kind == KIND_META:
            meta = rec.payload
            plain.clear()
        for msg_type, item in book_items(rec):
            sym = item["symbol"]
            if sym not in plain:
                p = meta["pairs"][sym]
                plain[sym] = L3Book(PairInfo(sym, p["price_precision"], p["qty_precision"]), meta["depth"])
            ok = plain[sym].apply(msg_type, item)
            expected[sym][{True: "checksum_ok", False: "checksum_fail", None: "skipped_unsynced"}[ok]] += 1
    for sym in symbols:
        got = result.counters[sym]
        assert {k: got[k] for k in expected[sym]} == dict(expected[sym])
        book = eng.state[sym].book
        assert (book.bids, book.asks, book.orders) == (plain[sym].bids, plain[sym].asks, plain[sym].orders)
        assert got["audit_fail"] == 0

    placed = [o for o in result.orders if o.inserted_ts is not None]
    assert placed and all(o.state is OrderState.ENDED and (o.end_reason or o.skip_reason) for o in result.orders)
    assert all(o.inserted_ts <= o.end_ts <= o.expiry_ts for o in placed)
    assert len(result.orders) + len(result.skips) == len(schedule)
    assert all(r.ahead_qty == 0 for r in result.reaches)
    seen = Counter((r.order_id, r.reach.ref) for r in result.reaches)
    assert max(seen.values(), default=1) == 1  # no historical event reaches the same order twice
    # every checksum failure is one incident, however many records describe it
    for sym in symbols:
        incidents = {q.incident for q in result.quality if q.symbol == sym and q.incident is not None}
        assert len(incidents) <= sum(q.symbol == sym and q.kind in DISCONTINUITY_KINDS for q in result.quality)
        assert result.counters[sym]["checksum_fail"] <= len(incidents)
    ends = Counter(o.end_reason.value for o in placed)
    diag = sum((o.diag for o in result.orders), Counter())
    print(f"\n{path.name}: {len(placed)} placed, {len(result.skips)} skipped, {len(result.reaches)} reaches "
          f"{dict(Counter(r.reach.mechanism.value for r in result.reaches))}, ends={dict(ends)}, "
          f"skips={dict(Counter(s.reason.value for s in result.skips))}, diag={dict(diag)}, "
          f"orders checksum_covered={dict(Counter(o.checksum_covered for o in placed))}, "
          f"reaches checksum_covered={dict(Counter(r.checksum_covered for r in result.reaches))}")


@pytest.mark.skipif(not L3_FILES, reason=f"no L3 recordings under {DATA_DIR / 'raw' / 'l3'}; run scripts/record.py")
def test_all_strategies_on_real_hour():
    path = min(L3_FILES, key=lambda p: p.stat().st_size)
    symbols = ("BTC/USD", "ETH/USD")
    start = int(file_hour(path, "l3").timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, start + 3600 * S, 10 * S)
    cfg = SimConfig(symbols=symbols, order_qty={s: "0.001" for s in symbols}, horizon_ns=10 * S,
                    audit_invariants=True)
    strategies = [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()]
    result = HistoricalReplayEngine(cfg, strategies, schedule).run([path])
    join_only = HistoricalReplayEngine(cfg, [JoinBestStrategy()], schedule).run([path])

    def summary(r, name):
        return ([(o.order_id, o.price, o.skip_reason, o.inserted_ts, o.end_reason, o.end_ts, o.queue_ahead_qty0,
                  o.max_level_rank) for o in r.orders if o.strategy == name],
                [(x.order_id, x.reach) for x in r.reaches if x.strategy == name],
                [(s.request.request_id, s.reason) for s in r.skips if s.strategy == name])

    assert summary(result, "join_best") == summary(join_only, "join_best")  # unchanged by the other strategies
    assert len(result.orders) + len(result.skips) == 3 * len(schedule)
    assert all(result.counters[s]["audit_fail"] == 0 for s in symbols)
    for o in result.orders:
        assert o.state is OrderState.ENDED and (o.end_reason or o.skip_reason)
        v = result.views[o.request.request_id]
        bid, ask = v.best_bid, v.best_ask
        assert o.price % v.tick == 0
        if o.strategy == "improve_quote":
            assert o.price == (bid + v.tick if o.side is Side.BUY else ask - v.tick)
        elif o.strategy == "quote_deeper":
            assert o.price == (bid - v.tick if o.side is Side.BUY else ask + v.tick)
        opposite = ask if o.side is Side.BUY else bid
        if opposite is not None and (o.strategy != "join_best" or bid < ask):  # join_best may copy a crossed touch
            assert o.price < opposite if o.side is Side.BUY else o.price > opposite
        if o.inserted_ts is not None:
            assert o.inserted_ts <= o.end_ts <= o.expiry_ts
    for name in ("join_best", "improve_quote", "quote_deeper"):
        placed = [o for o in result.orders if o.strategy == name and o.inserted_ts is not None]
        print(f"\n{name}: {len(placed)} placed, "
              f"skips={dict(Counter(s.reason.value for s in result.skips if s.strategy == name))}, "
              f"reaches={dict(Counter(r.reach.mechanism.value for r in result.reaches if r.strategy == name))}, "
              f"ends={dict(Counter(o.end_reason.value for o in placed))}")
    assert any(o.strategy == "quote_deeper" and o.inserted_ts is not None for o in result.orders)


EVAL_FILE = next((p for p in L3_FILES if p.name == "l3_2026-10-18T14.jsonl.gz"), None)


@pytest.mark.skipif(EVAL_FILE is None, reason="l3_2026-10-18T14.jsonl.gz not recorded")
def test_metrics_on_real_hour():
    symbols = ("BTC/USD", "ETH/USD")
    start = int(file_hour(EVAL_FILE, "l3").timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, start + 3600 * S, 10 * S)
    cfg = SimConfig(symbols=symbols, order_qty={s: "0.001" for s in symbols}, horizon_ns=10 * S)
    strategies = [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()]
    names = [s.name for s in strategies]
    result = HistoricalReplayEngine(cfg, strategies, schedule, on_reach=FillRecorder()).run([EVAL_FILE])
    mids = sample_mids(cfg, required_mid_times(result, cfg), files=[EVAL_FILE])
    report = analyze(result, cfg, mids, strategies=names)
    assert analyze(result, cfg, mids, strategies=names) == report  # deterministic aggregation

    # request accounting reconciles with the engine
    rc = report.request_counts
    assert rc["total"] == len(schedule)
    assert rc["total"] == rc["eligible"] + sum(rc["data_gate"].values()) + sum(rc["coverage"].values())
    gated = {s.request.request_id for s in result.skips if s.request.request_id not in result.views}
    assert sum(rc["data_gate"].values()) == len(gated)
    for name in names:
        lc = report.lifecycle[name]
        assert lc.eligible == rc["eligible"] == lc.strategy_skips + lc.orders_created
        assert lc.orders_created == lc.insertion_skips + lc.placed
        assert lc.placed == lc.expired + lc.ended_all_filled + lc.censored + lc.censored_after_horizon
        engine_placed = sum(o.strategy == name and o.inserted_ts is not None for o in result.orders)
        assert lc.placed <= engine_placed
    # every fill counted once, never above the order size, policies ordered (I6)
    keys = [(f.policy, f.order_id, f.event_ref) for f in report.fills]
    assert len(keys) == len(set(keys))
    settled = Counter()
    for f in report.fills:
        settled[(f.policy, f.order_id)] += f.qty
    assert all(q <= D("0.001") for q in settled.values())
    for o in report.orders:
        if o.outcomes:
            got = [o.outcomes[p].filled_qty for p in ("conservative", "balanced", "optimistic")]
            assert got == sorted(got) and got[0] == 0  # L3-only conservative never fills
    # rates stay in [0, 1]; markout signs agree with the sampled mids; missing is never zero
    for s in report.summary.values():
        for r in (s.fill_given_placed, s.fill_per_opportunity, s.fill_common_support, s.full_fill_given_placed):
            assert r.value is None or 0 <= r.value <= 1
        for m in s.markouts.values():
            assert m.n_valid + sum(m.missing.values()) == m.n_fills
    for f in report.fills:
        for d, v in f.markout_bps.items():
            sample = mids[(f.symbol, f.ts_ns + d)]
            assert (v is None) == (sample.mid is None) and (v is None) == (f.markout_missing[d] is not None)
            if v is not None:
                assert (v > 0) == (f.side.sign * (sample.mid - f.price) > 0)
    # placement frequency differs: improve_quote is conditional on a 2-tick spread
    assert report.lifecycle["improve_quote"].placed < report.lifecycle["join_best"].placed
    print("\n" + format_report(report, markout_horizons_ns=(S, 5 * S, 10 * S)))


PUBLIC_FILE = next((p for p in list_files(DATA_DIR / "raw", "public") if p.name == "public_2026-10-18T14.jsonl.gz"),
                   None)


@pytest.mark.skipif(EVAL_FILE is None or PUBLIC_FILE is None, reason="hour 14 L3 + public recordings not present")
def test_trade_attribution_on_real_hour():
    symbols = ("BTC/USD", "ETH/USD")
    start = int(file_hour(EVAL_FILE, "l3").timestamp()) * S
    schedule = PlacementSchedule.regular(symbols, start, start + 3600 * S, 10 * S)
    cfg = SimConfig(symbols=symbols, order_qty={s: "0.001" for s in symbols}, horizon_ns=10 * S)
    names = ["join_best", "improve_quote", "quote_deeper"]
    index = TradeIndex.from_files([PUBLIC_FILE], symbols, gap_ns=cfg.gap_ns)
    att = attribute_trades(cfg, index, files=[EVAL_FILE])
    result = HistoricalReplayEngine(cfg, [JoinBestStrategy(), ImproveQuoteStrategy(), QuoteDeeperStrategy()],
                                    schedule, on_reach=FillRecorder(att)).run([EVAL_FILE])
    mids = sample_mids(cfg, required_mid_times(result, cfg, evidence=("l3_only", "trades")), files=[EVAL_FILE])
    l3 = analyze(result, cfg, mids, strategies=names)
    tr = analyze(result, cfg, mids, strategies=names, evidence="trades", attribution=att)

    # every print is accounted for exactly once, and no print's volume is used twice
    n_trades = sum(len(v) for v in index.trades.values())
    assert sum(att.stats["trades"].values()) == n_trades
    by_ref = {(sym, t.ref): t.qty for sym, ts in index.trades.items() for t in ts}
    assert all(0 <= used <= by_ref[k] for k, used in att.trade_used.items())
    attributed = sum((p.qty for r in att.by_ref.values() for p in r.parts
                      if p.evidence in (Evidence.TRADE_CONFIRMED, Evidence.TRADE_PLAUSIBLE)), D(0))
    assert attributed == sum(att.trade_used.values())
    matched = sum(n for k, n in att.stats["trades"].items() if k.endswith("|MATCHED"))
    assert matched >= 0.99 * n_trades, att.stats["trades"]  # measured: 1524 of 1527 on this hour

    # placement and queue do not depend on evidence; trade-mode fills are bounded by the L3-only upper bound
    assert l3.lifecycle == tr.lifecycle and l3.queue == tr.queue and l3.request_counts == tr.request_counts
    order_by_id = {o.order_id: o for o in tr.orders if o.outcomes}
    for o in l3.orders:
        if not o.outcomes:
            continue
        t = order_by_id[o.order_id].outcomes
        got = [t[p].filled_qty for p in ("conservative", "balanced", "optimistic")]
        assert got == sorted(got) and got[2] <= o.outcomes["optimistic"].filled_qty and got[2] <= o.qty
    allowed = {"conservative": {"TRADE_CONFIRMED"}, "balanced": {"TRADE_CONFIRMED", "TRADE_PLAUSIBLE"}}
    keys = [(f.policy, f.order_id, f.event_ref) for f in tr.fills]
    assert len(keys) == len(set(keys))
    for f in tr.fills:
        assert "TRADE_ABSENT" not in f.qty_by_evidence and sum(f.qty_by_evidence.values()) == f.qty
        assert set(f.qty_by_evidence) <= allowed.get(f.policy, set(f.qty_by_evidence))
    print("\n" + format_report(l3, markout_horizons_ns=(S, 10 * S)) + "\n\n" +
          format_report(tr, markout_horizons_ns=(S, 10 * S)))
