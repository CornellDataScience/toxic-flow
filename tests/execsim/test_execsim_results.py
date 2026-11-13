"""ResultWriter and scripts/run_simulation.py end to end on synthetic recordings (design §11.1 to §11.3).

Every run goes through the CLI's main(), so these tests cover the whole path: reader, engine,
strategies, queue tracking, settlement, trade attribution, metrics and the written files.
"""

import hashlib
import importlib.util
import json
from decimal import Decimal
from pathlib import Path

import pytest
from execsim_helpers import order
from test_execsim_crossing import three_order_tape
from test_execsim_replay import Tape, at
from test_execsim_trades import Public

from toxicflow.execsim.results import DATA_FILES, ResultWriter, dumps_row, jsonable, stable_hash
from toxicflow.execsim.timeutil import format_ns

D = Decimal
REPO = Path(__file__).resolve().parents[2]
POLICIES = ("conservative", "balanced", "optimistic")
SYM = "BTC/USD"

_spec = importlib.util.spec_from_file_location("run_simulation", REPO / "scripts" / "run_simulation.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


def run_cli(l3, out, *extra, start_ms=5, end_ms=6500, capsys=None):
    argv = [*map(str, l3), "--symbols", SYM, "--qty", "0.5", "--interval-s", "2", "--horizon-s", "1",
            "--start", format_ns(at(start_ms)), "--end", format_ns(at(end_ms)), "--out", str(out), *extra]
    run_dir = cli.main(argv)
    text = capsys.readouterr().out if capsys else None
    return run_dir, text


def rows(run_dir, name):
    return [json.loads(line) for line in (run_dir / name).read_text().splitlines()]


def manifest(run_dir):
    return json.loads((run_dir / "manifest.json").read_text())


def by_order(run_dir):
    return {r["order_id"]: r for r in rows(run_dir, "orders.jsonl")}


def trade_public():
    pub = Public(cover=(0, 7_000))
    pub.trade(at(40), "sell", "100.0", "0.6")  # behind order 1's fill
    pub.trade(at(2040), "sell", "99.9", "0.4")  # the print after order 2 was crossed
    return pub


# --- files, schema and links ---------------------------------------------------------------------


def test_run_writes_every_file_with_linked_rows(tmp_path, capsys):
    l3 = three_order_tape().write(tmp_path / "raw")
    run_dir, text = run_cli(l3, tmp_path / "sim", capsys=capsys)
    assert sorted(p.name for p in run_dir.iterdir()) == sorted([*DATA_FILES, "manifest.json"])
    man, orders, fills = manifest(run_dir), rows(run_dir, "orders.jsonl"), rows(run_dir, "fills.jsonl")
    schedule, quality = rows(run_dir, "schedule.jsonl"), rows(run_dir, "quality.jsonl")

    # one order row per (request, strategy); ids are unique and link every file
    assert len(schedule) == 8 and len(orders) == 8 * 3 == man["files"]["orders.jsonl"]["rows"]
    assert {r["request_id"] for r in orders} == {r["request_id"] for r in schedule}
    ids = [r["order_id"] for r in orders]
    assert len(set(ids)) == len(ids) and all(r["order_id"] == f"{r['request_id']}/{r['strategy']}" for r in orders)
    assert all(r["run_id"] == man["run_id"] == run_dir.name for r in [*orders, *fills, *quality])
    placed = {r["order_id"]: r for r in orders if r["placed"]}
    assert fills and {f["order_id"] for f in fills} <= set(placed)
    for oid, r in placed.items():
        mine = [f for f in fills if f["order_id"] == oid]
        assert len(mine) == r["n_candidates"] and len({tuple(f["event_ref"]) for f in mine}) == len(mine)
        for p in POLICIES:  # what a policy accepted of the candidates is exactly the order's fill
            assert sum((D(f[f"accepted_qty_{p}"]) for f in mine), D(0)) == D(r[f"filled_qty_{p}"])
            assert D(r[f"filled_qty_{p}"]) + D(r[f"remaining_qty_{p}"]) == D(r["order_qty"])
        assert all(D(f["fill_price"]) == D(r["limit_price"]) and f["strategy"] == r["strategy"] for f in mine)
        assert all(r["placement_ts_ns"] <= f["fill_ts_ns"] <= r["end_ts_ns"] for f in mine)

    # decimals are strings that round-trip exactly; times are integer nanoseconds
    first = by_order(run_dir)["BTC/USD|buy|" + str(at(5)) + "/join_best"]
    assert (first["limit_price"], first["order_qty"], first["queue_ahead_qty"], first["tick_size"]) == (
        "100.0", "0.5", "3.0", "0.1")
    assert (first["decision_ts_ns"], first["placement_ts_ns"], first["end_ts_ns"]) == (at(5), at(5), at(1005))
    assert (first["queue_ahead_orders"], first["level_rank"], first["horizon_ns"]) == (2, 1, 1_000_000_000)
    assert (first["best_bid_at_decision"], first["best_ask_at_decision"]) == ("100.0", "100.2")
    assert first["end_reason"] == "HORIZON" and first["censored"] is False and first["checksum_covered"] is True
    assert (first["status_balanced"], first["filled_qty_balanced"]) == ("FILLED", "0.5")
    assert first["time_to_first_fill_ns_balanced"] == first["time_to_complete_fill_ns_balanced"] == 35_000_000
    assert first["status_conservative"] == "UNFILLED" and first["fill_model"].startswith("l3_only,cross_rule=stop")
    for r in orders:
        for key in ("limit_price", "order_qty", "queue_ahead_qty", "touch_qty", "cross_price", "cross_qty"):
            assert r[key] is None or isinstance(r[key], str), key
        for key in ("decision_ts_ns", "placement_ts_ns", "end_ts_ns", "cross_ts_ns"):
            assert r[key] is None or isinstance(r[key], int), key
    raw = (run_dir / "orders.jsonl").read_text() + (run_dir / "fills.jsonl").read_text()
    assert "e-" not in raw.lower() and "E+" not in raw  # no float or exponent notation anywhere

    # candidates: the level fill of order 1, with its source line in the recording
    (f,) = [f for f in fills if f["order_id"] == first["order_id"]]
    assert (f["mechanism"], f["evidence"], f["evidence_rank"], f["fill_qty"], f["fill_ts_ns"]) == (
        "level", "L3_MODIFY_DOWN", 3, "0.6", at(40))
    assert (f["accepted_qty_balanced"], f["accepted_qty_conservative"], f["ahead_qty_at_fill"]) == ("0.5", "0", "0")
    assert f["source_file"] == l3[0].name and f["source_line"] == 5 and f["trade_evidence"] is None
    assert f["in_horizon"] and not f["after_cross"] and not f["is_cross_event"]

    # quality events and the manifest
    assert [q["kind"] for q in quality] == ["SESSION_RESTART", "SNAPSHOT_FIRST", "END_OF_DATA"]
    assert quality[0]["detail"] == {"file": l3[0].name} and quality[1]["line_no"] == 2
    assert man["schema_version"] == 1 and man["run"]["evidence_mode"] == "l3_only"
    assert man["run"]["cross_rule"] == "stop" and man["config"]["order_qty"] == {SYM: "0.5"}
    assert man["schedule"]["n_requests"] == 8 and man["inputs"]["trades"] == []
    assert man["inputs"]["l3"][0]["sha256"] == hashlib.sha256(l3[0].read_bytes()).hexdigest()
    assert set(man["code"]) == {"commit", "branch", "dirty", "changed_files"}
    assert man["summary"]["strategies"]["join_best"]["placed"] == 6
    assert man["quality"]["incidents"] == {SYM: 1}
    for name in DATA_FILES:
        assert man["files"][name]["sha256"] == hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
    assert {"started_utc", "wall_seconds", "argv"} <= set(man["runtime"])

    # the printed summary
    assert "L3-ONLY" in text and "trade-confirmed" in text and f"output: {run_dir}" in text
    assert "cross rule stop" in text and "requests 8: eligible 6" in text and "data quality" in text


def test_skipped_gated_and_censored_orders(tmp_path):
    tape = Tape()
    tape.update(2, bids=[order("X1", "99.0", "1.0", "add")])  # before the first snapshot: unsynced
    tape.snapshot(10, bids=[order("B1", "100.0", "1.0")], asks=[order("A1", "100.1", "3.0")])  # one-tick spread
    tape.update(3_000, bids=[order("B2", "100.0", "1.0", "add")])
    tape.update(4_500, bids=[order("B2", "100.0", "1.0", "delete")])
    run_dir, _ = run_cli(tape.write(tmp_path / "raw"), tmp_path / "sim", "--sides", "buy", end_ms=4_100)
    got = {(r["decision_ts_ns"], r["strategy"]): r for r in rows(run_dir, "orders.jsonl")}

    gated = got[(at(5), "join_best")]  # the data gate refused the decision for every strategy
    assert (gated["placed"], gated["skip_reason"], gated["skip_stage"]) == (False, "UNSYNCED", "decision")
    assert (gated["request_status"], gated["eligible"], gated["limit_price"]) == ("DATA_GATE", False, None)
    assert gated["status_balanced"] is None and gated["filled_qty_balanced"] is None and gated["order_qty"] == "0.5"

    skipped = got[(at(2_005), "improve_quote")]  # a strategy skip on an eligible request
    assert (skipped["placed"], skipped["skip_reason"], skipped["eligible"]) == (False, "SPREAD_TOO_NARROW", True)
    assert skipped["best_bid_at_decision"] == "100.0" and skipped["n_candidates"] is None
    ok = got[(at(2_005), "join_best")]
    assert (ok["placed"], ok["end_reason"], ok["censored"], ok["status_balanced"]) == (True, "HORIZON", False, "UNFILLED")

    cut = got[(at(4_005), "join_best")]  # the data ends before its horizon: censored, never a non-fill
    assert (cut["placed"], cut["end_reason"], cut["censored"]) == (True, "CENSORED_END_OF_DATA", True)
    assert (cut["request_status"], cut["request_status_reason"], cut["eligible"]) == ("COVERAGE", "END_OF_DATA", False)
    summary = manifest(run_dir)["summary"]
    assert summary["requests"]["eligible"] == 1 and summary["requests"]["data_gate"] == {"UNSYNCED": 1}
    assert summary["strategies"]["improve_quote"]["strategy_skip_reasons"] == {"SPREAD_TOO_NARROW": 1}


# --- reproducibility --------------------------------------------------------------------------------


def test_rerun_is_byte_identical(tmp_path, capsys):
    root = tmp_path / "raw"
    l3 = three_order_tape().write(root)
    trade_public().write(root)
    a, _ = run_cli(l3, tmp_path / "a", "--trades", "auto")
    b, _ = run_cli(l3, tmp_path / "b", "--trades", "auto")
    assert a.name == b.name  # the run id is derived from the inputs and settings
    for name in DATA_FILES:
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ma, mb = manifest(a), manifest(b)
    assert ma.pop("runtime")["argv"] != mb.pop("runtime")["argv"]  # only the volatile block differs
    assert ma == mb

    with pytest.raises(SystemExit, match="already holds a run"):
        run_cli(l3, tmp_path / "a", "--trades", "auto")
    before = {name: (a / name).read_bytes() for name in DATA_FILES}
    (a / "notes.txt").write_text("mine")
    run_cli(l3, tmp_path / "a", "--trades", "auto", "--overwrite")
    assert {name: (a / name).read_bytes() for name in DATA_FILES} == before
    assert (a / "notes.txt").read_text() == "mine"  # only the writer's own files are replaced

    other, _ = run_cli(l3, tmp_path / "a", "--trades", "auto", "--horizon-s", "0.5")  # another setting, another run
    assert other.name != a.name and manifest(other)["config_hash"] != ma["config_hash"]
    named, _ = run_cli(l3, tmp_path / "a", "--run-id", "exp1")
    assert named.name == "exp1" and rows(named, "orders.jsonl")[0]["run_id"] == "exp1"


# --- evidence modes -----------------------------------------------------------------------------------


def test_l3_only_and_trade_evidence_runs(tmp_path, capsys):
    root = tmp_path / "raw"
    l3 = three_order_tape().write(root)
    pub = trade_public().write(root)
    plain, text_plain = run_cli(l3, tmp_path / "sim", capsys=capsys)
    traded, text_traded = run_cli(l3, tmp_path / "sim", "--trades", *map(str, pub), capsys=capsys)
    assert plain.name != traded.name and "_l3_only_" in plain.name and "_trades_" in traded.name

    fills = rows(plain, "fills.jsonl")
    assert all(f["trade_evidence"] is None and f["trade_ids"] == [] for f in fills)
    assert not any(f["evidence"].startswith("TRADE") for f in fills)
    assert all(r["status_conservative"] in (None, "UNFILLED") for r in rows(plain, "orders.jsonl"))
    assert "TRADE_" not in (plain / "orders.jsonl").read_text() and "L3-ONLY" in text_plain
    assert manifest(plain)["trade_attribution"] is None

    man = manifest(traded)
    assert man["run"]["evidence_mode"] == "trades" and [f["name"] for f in man["inputs"]["trades"]] == [pub[0].name]
    assert man["trade_attribution"]["trade_feed"]["trades"] == 2
    key = "BTC/USD|buy|" + str(at(5)) + "/join_best"
    order1 = by_order(traded)[key]
    assert order1["fill_model"].startswith("l3_trades") and order1["max_evidence_rank"] == 5
    assert (order1["status_conservative"], order1["filled_qty_conservative"]) == ("FILLED", "0.5")
    (f,) = [f for f in rows(traded, "fills.jsonl") if f["order_id"] == key]
    assert f["evidence"] == "L3_MODIFY_DOWN"  # the L3 label of the candidate is never rewritten
    assert f["trade_evidence"] == [{"qty": "0.6", "evidence": "TRADE_CONFIRMED", "evidence_rank": 5,
                                    "trade_ids": [1000]}] and f["trade_ids"] == [1000]
    assert f["accepted_qty_conservative"] == "0.5" and "L3 + trade prints" in text_traded

    with pytest.raises(SystemExit):  # asking for trade evidence without recordings is an error, not a fallback
        run_cli(l3, tmp_path / "sim", "--evidence", "trades")
    capsys.readouterr()
    with pytest.raises(SystemExit, match="not found"):
        run_cli(l3, tmp_path / "sim", "--trades", str(tmp_path / "missing.jsonl.gz"))
    forced, _ = run_cli(l3, tmp_path / "sim2", "--trades", "auto", "--evidence", "l3_only")
    assert manifest(forced)["run"]["evidence_mode"] == "l3_only" and manifest(forced)["inputs"]["trades"] == []


# --- crossing rules --------------------------------------------------------------------------------------


def test_crossing_rules_change_settlement_but_never_the_candidates(tmp_path):
    root = tmp_path / "raw"
    l3 = three_order_tape().write(root)
    trade_public().write(root)
    runs = {rule: run_cli(l3, tmp_path / "sim", "--trades", "auto", "--cross-rule", rule)[0]
            for rule in ("ignore", "stop", "fill_at_cross")}
    key = "BTC/USD|buy|" + str(at(2005)) + "/join_best"  # crossed at 2030, traded through at 2040

    def raw(run_dir):  # the historical part of every candidate row
        return [{k: v for k, v in f.items() if k != "run_id" and not k.startswith("accepted_qty_")}
                for f in rows(run_dir, "fills.jsonl")]

    assert raw(runs["ignore"]) == raw(runs["stop"]) == raw(runs["fill_at_cross"])
    assert (runs["ignore"] / "quality.jsonl").read_text().replace(runs["ignore"].name, "") == (
        runs["stop"] / "quality.jsonl").read_text().replace(runs["stop"].name, "")

    crossed = {rule: by_order(d)[key] for rule, d in runs.items()}
    for r in crossed.values():  # the crossing itself is a fact of the recording
        assert (r["crossed"], r["cross_ts_ns"], r["cross_price"], r["cross_qty"], r["cross_ahead_qty"]) == (
            True, at(2030), "100.0", "0.2", "0")
        assert (r["n_crossings"], r["inconsistent_crossing"], r["cross_event_kind"]) == (1, True, "add")
    assert (crossed["ignore"]["status_balanced"], crossed["ignore"]["filled_qty_balanced"]) == ("PARTIAL", "0.4")
    assert crossed["ignore"]["resolved_by_cross_balanced"] is False
    for rule in ("stop", "fill_at_cross"):
        r = crossed[rule]
        assert (r["status_balanced"], r["filled_qty_balanced"], r["resolved_by_cross_balanced"]) == (
            "UNFILLED", "0", True)
        assert r["first_fill_ts_ns_balanced"] is None and f"cross_rule={rule}" in r["fill_model"]
    assert (crossed["stop"]["status_optimistic"], crossed["fill_at_cross"]["status_optimistic"]) == (
        "UNFILLED", "FILLED")

    late = {rule: [f for f in rows(d, "fills.jsonl") if f["order_id"] == key] for rule, d in runs.items()}
    for rule, (arrival, trade) in late.items():
        assert (arrival["is_cross_event"], arrival["after_cross"], arrival["mechanism"]) == (True, False, "cross_arrival")
        assert (trade["after_cross"], trade["trade_evidence"][0]["evidence"]) == (True, "TRADE_CONFIRMED")
        assert trade["accepted_qty_balanced"] == ("0.4" if rule == "ignore" else "0")

    # an uncrossed order is the same under every rule, and the manifest counts what the rule changed
    same = "BTC/USD|buy|" + str(at(5)) + "/join_best"
    assert len({json.dumps({k: v for k, v in by_order(d)[same].items() if k not in ("run_id", "fill_model")})
                for d in runs.values()}) == 1
    bal = {rule: manifest(d)["summary"]["strategies"]["join_best"]["policies"]["balanced"] for rule, d in runs.items()}
    assert [bal[rule]["filled_any"] for rule in ("ignore", "stop", "fill_at_cross")] == [2, 1, 1]
    assert (bal["stop"]["crossed_unfilled"], bal["stop"]["legacy_filled_any"],
            bal["stop"]["legacy_first_fill_after_cross"], bal["stop"]["fill_given_placed"]) == (1, 2, 1, [1, 5])
    assert bal["ignore"]["fill_given_placed"] == [2, 6]  # 6 placed (buys and sells); one outcome unknown under stop
    assert manifest(runs["stop"])["summary"]["strategies"]["join_best"]["crossed"] == 1


# --- several files -----------------------------------------------------------------------------------------


def test_multi_file_replay_in_hour_order(tmp_path):
    tape = three_order_tape()
    tape.new_file()  # an hourly rotation: the next file starts with stale updates, then a snapshot
    tape.update(6_600, asks=[order("A8", "100.8", "1.0", "add")])
    book = tape.sealer.book
    tape.snapshot(6_700, bids=[order(oid, str(px), str(q)) for px, lvl in book.bids.items() for oid, q in lvl.items()],
                  asks=[order(oid, str(px), str(q)) for px, lvl in book.asks.items() for oid, q in lvl.items()])
    tape.update(8_100, bids=[order("C1", "99.9", "4.0", "modify")])
    tape.update(9_900, asks=[order("A8", "100.8", "1.0", "delete")])
    l3 = tape.write(tmp_path / "raw")
    assert len(l3) == 2
    run_dir, _ = run_cli(reversed(l3), tmp_path / "sim", "--sides", "buy", end_ms=9_000)  # given out of order
    man = manifest(run_dir)
    assert [f["name"] for f in man["inputs"]["l3"]] == [p.name for p in l3]
    quality = rows(run_dir, "quality.jsonl")
    assert [q["kind"] for q in quality] == ["SESSION_RESTART", "SNAPSHOT_FIRST", "SESSION_RESTART", "RESNAPSHOT",
                                            "END_OF_DATA"]
    assert [q["file"] for q in quality[:4]] == [l3[0].name, l3[0].name, l3[1].name, l3[1].name]
    assert man["quality"]["incidents"] == {SYM: 2} and man["quality"]["counters"][SYM]["skipped_unsynced"] == 1

    got = {r["decision_ts_ns"]: r for r in rows(run_dir, "orders.jsonl") if r["strategy"] == "join_best"}
    live = got[at(6_005)]  # live across the file boundary: censored there, and excluded by the coverage filter
    assert (live["placed"], live["end_reason"], live["censored"]) == (True, "CENSORED_SESSION_RESTART", True)
    assert (live["request_status"], live["eligible"]) == ("COVERAGE", False)
    after = got[at(8_005)]  # placed on the second file's book, behind C1, and filled by its reduction
    assert (after["placed"], after["eligible"], after["limit_price"], after["queue_ahead_qty"]) == (
        True, True, "99.9", "4.6")
    fills = [f for f in rows(run_dir, "fills.jsonl") if f["order_id"] == after["order_id"]]
    assert fills == [] and after["status_balanced"] == "UNFILLED"  # C1 was ahead: its reduction only advances us
    default_range, _ = run_cli(l3, tmp_path / "sim", "--sides", "buy", "--run-id", "whole")
    whole = manifest(default_range)["schedule"]
    assert (whole["start_ns"], whole["end_ns"]) == (at(5), at(6500))  # run_cli's own range


def test_default_schedule_covers_the_files_and_default_symbols_come_from_meta(tmp_path):
    l3 = three_order_tape().write(tmp_path / "raw")
    run_dir = cli.main([str(l3[0]), "--qty", "BTC/USD=0.5", "ETH/USD=2", "--interval-s", "1800", "--horizon-s", "1",
                        "--strategies", "join_best", "quote_deeper", "--depth-ticks", "3", "--out",
                        str(tmp_path / "sim")])
    man = manifest(run_dir)
    assert man["config"]["symbols"] == ["BTC/USD", "ETH/USD"] and man["config"]["order_qty"]["ETH/USD"] == "2"
    assert (man["schedule"]["start_ns"], man["schedule"]["end_ns"]) == (at(0), at(3_600_000))
    assert man["schedule"]["n_requests"] == 2 * 2 * 2
    assert man["run"]["strategies"] == [{"name": "join_best", "class": "JoinBestStrategy", "params": {}},
                                        {"name": "quote_deeper", "class": "QuoteDeeperStrategy",
                                         "params": {"depth_ticks": 3}}]
    assert {r["skip_reason"] for r in rows(run_dir, "orders.jsonl") if r["symbol"] == "ETH/USD"} == {"NO_DATA"}


# --- writer units --------------------------------------------------------------------------------------------


def test_serialization_is_exact_and_stable(tmp_path):
    assert dumps_row({"p": D("81846.20"), "q": D("1E-8"), "t": 1792296000663208000, "s": (1, 2)}) == (
        '{"p":"81846.20","q":"0.00000001","t":1792296000663208000,"s":[1,2]}')
    with pytest.raises(TypeError):
        dumps_row({"p": 0.1})  # a float can never reach a result row
    assert jsonable({"x": 0.5}, strict=False) == {"x": 0.5}
    assert stable_hash({"a": D("1.0"), "b": [1, 2]}) == stable_hash({"b": (1, 2), "a": D("1.0")})
    assert stable_hash({"a": D("1.0")}) != stable_hash({"a": D("1.00")})

    w = ResultWriter(tmp_path / "run", "r1")
    assert w.write_manifest({"runtime": {"wall_seconds": 0.25}, "x": D("2.5")}).exists()
    assert manifest(tmp_path / "run") == {"schema_version": 1, "run_id": "r1", "x": "2.5", "files": {},
                                          "runtime": {"wall_seconds": 0.25}}
    with pytest.raises(FileExistsError):
        ResultWriter(tmp_path / "run", "r1")
    ResultWriter(tmp_path / "run", "r1", overwrite=True)
    assert not (tmp_path / "run" / "manifest.json").exists()
