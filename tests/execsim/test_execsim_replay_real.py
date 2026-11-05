"""Real-data smoke run of the replay engine (Phase 1 acceptance checks that need no fill model)."""

from collections import Counter

import pytest

from toxicflow.execsim.config import SimConfig
from toxicflow.execsim.orders import OrderState
from toxicflow.execsim.reader import KIND_META, RecordingReader, book_items, file_hour, list_files
from toxicflow.execsim.replay import HistoricalReplayEngine
from toxicflow.execsim.results import DISCONTINUITY_KINDS
from toxicflow.execsim.schedule import PlacementSchedule
from toxicflow.execsim.strategies import JoinBestStrategy
from toxicflow.execsim.timeutil import S
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
