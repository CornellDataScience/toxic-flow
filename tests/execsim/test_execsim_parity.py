"""SimL3Book must reconstruct exactly what L3Book does on real data."""

from collections import Counter

import pytest

from toxicflow.execsim.reader import KIND_META, RecordingReader, book_items, list_files
from toxicflow.execsim.sim_book import SimL3Book
from toxicflow.kraken.book import L3Book, PairInfo
from toxicflow.kraken.config import DATA_DIR

L3_FILES = list_files(DATA_DIR / "raw", "l3")


@pytest.mark.skipif(not L3_FILES, reason=f"no L3 recordings under {DATA_DIR / 'raw' / 'l3'}; run scripts/record.py")
def test_replay_parity_with_l3book():
    path = min(L3_FILES, key=lambda p: p.stat().st_size)
    plain: dict[str, L3Book] = {}
    sim: dict[str, SimL3Book] = {}
    counts: dict[str, Counter] = {"plain": Counter(), "sim": Counter()}
    mismatches = []
    n_events = 0
    for rec in RecordingReader(files=[path]):
        if rec.kind == KIND_META:
            meta = rec.payload
            plain.clear()
            sim.clear()
            continue
        for msg_type, item in book_items(rec):
            sym = item["symbol"]
            if sym not in plain:
                p = meta["pairs"][sym]
                pair = PairInfo(sym, p["price_precision"], p["qty_precision"])
                plain[sym], sim[sym] = L3Book(pair, meta["depth"]), SimL3Book(pair, meta["depth"])
            a, b = plain[sym].apply(msg_type, item), sim[sym].apply(msg_type, item, position=rec)
            counts["plain"][(msg_type, a)] += 1
            counts["sim"][(msg_type, b)] += 1
            n_events += len(sim[sym].last_events)
            top_a = (plain[sym].best_bid(), plain[sym].best_ask())
            top_b = (sim[sym].best_bid(), sim[sym].best_ask())
            if a != b or top_a != top_b:
                mismatches.append((rec.seq, sym, a, b, top_a, top_b))
    assert not mismatches[:10]
    assert counts["sim"] == counts["plain"] and sum(counts["sim"].values()) > 0
    for sym in plain:
        assert sim[sym].bids == plain[sym].bids and sim[sym].asks == plain[sym].asks
        assert sim[sym].orders == plain[sym].orders
        assert sim[sym].event_counts == plain[sym].event_counts
    print(f"\n{path.name}: {dict(counts['sim'])}, {n_events} events")
