from datetime import datetime, timezone
from decimal import Decimal

from execsim_helpers import gz_member, recording_lines, write_file

from toxicflow.execsim.reader import KIND_GAP, KIND_META, KIND_MSG, KIND_TRUNCATED, RecordingReader, book_items
from toxicflow.kraken.recorder import read_recording

S = 1_000_000_000
HB = {"channel": "heartbeat"}


def kinds(records):
    return [r.kind for r in records]


def test_files_in_hour_order_with_meta_and_positions(tmp_path):
    # written out of order and across a date boundary
    b = write_file(tmp_path, "2026-10-19T00", gz_member(recording_lines([(10 * S, HB)])))
    a = write_file(tmp_path, "2026-10-18T23", gz_member(recording_lines([(1 * S, HB), (2 * S, HB)])))
    recs = list(RecordingReader(tmp_path))
    assert [r.file for r in recs] == [a, a, a, b, b, b]
    assert kinds(recs) == [KIND_META, KIND_MSG, KIND_MSG, KIND_META, KIND_GAP, KIND_MSG]  # 8 s apart across files
    assert [r.seq for r in recs] == list(range(len(recs)))
    assert recs[0].is_meta and recs[0].line_no == 1 and recs[0].payload["depth"] == 10
    assert [r.line_no for r in recs if r.file == a] == [1, 2, 3]


def test_matches_read_recording_on_clean_file(tmp_path):
    msgs = [(1 * S, HB), (2 * S, {"channel": "trade", "type": "update", "data": [{"price": Decimal("1.50")}]})]
    path = write_file(tmp_path, "2026-10-18T02", gz_member(recording_lines(msgs)) * 2)  # two gzip members
    ours = [(r.recv_ns, r.kind, r.payload) for r in RecordingReader(files=[path], gap_ns=10**18)]
    assert ours == list(read_recording(path))
    assert [k for _, k, _ in ours] == [KIND_META, KIND_MSG, KIND_MSG] * 2


def test_floats_are_decimal_ints_stay_int(tmp_path):
    raw = gz_member(['{"recv_ns":0,"meta":{}}', '{"recv_ns":1,"msg":{"price":81555.10,"trade_id":7}}'])
    msg = list(RecordingReader(files=[write_file(tmp_path, "2026-10-18T02", raw)]))[1].payload
    assert msg["price"] == Decimal("81555.10") and type(msg["price"]) is Decimal
    assert type(msg["trade_id"]) is int


def test_gap_marker_before_the_late_record(tmp_path):
    msgs = [(1 * S, HB), (2 * S, HB), (8 * S, HB), (13 * S, HB)]  # 6 s gap, then exactly 5 s (not a gap)
    recs = list(RecordingReader(files=[write_file(tmp_path, "2026-10-18T02", gz_member(recording_lines(msgs)))]))
    assert kinds(recs) == [KIND_META, KIND_MSG, KIND_MSG, KIND_GAP, KIND_MSG, KIND_MSG]
    gap = recs[3]
    assert gap.payload == {"gap_ns": 6 * S, "prev_recv_ns": 2 * S}
    assert gap.recv_ns == 8 * S and gap.line_no == recs[4].line_no
    assert [r.seq for r in recs] == list(range(6))

    tight = list(RecordingReader(files=[recs[0].file], gap_ns=S // 2))
    assert kinds(tight).count(KIND_GAP) == 3


def test_gap_detected_across_files(tmp_path):
    write_file(tmp_path, "2026-10-18T02", gz_member(recording_lines([(1 * S, HB)])))
    write_file(tmp_path, "2026-10-18T04", gz_member(recording_lines([(7300 * S, HB)])))
    assert kinds(RecordingReader(tmp_path)) == [KIND_META, KIND_MSG, KIND_META, KIND_GAP, KIND_MSG]


def test_truncated_tail_marks_and_continues(tmp_path):
    good = gz_member(recording_lines([(1 * S, HB), (2 * S, HB)]))
    write_file(tmp_path, "2026-10-18T02", good[:-8])  # gzip trailer missing
    write_file(tmp_path, "2026-10-18T03", good)
    reader = RecordingReader(tmp_path, gap_ns=10**18)
    recs = list(reader)
    assert kinds(recs) == [KIND_META, KIND_MSG, KIND_MSG, KIND_TRUNCATED, KIND_META, KIND_MSG, KIND_MSG]
    assert recs[3].payload["error"].startswith("EOFError") and recs[3].recv_ns == 2 * S
    assert reader.stats[KIND_TRUNCATED] == 1


def test_truncated_member_followed_by_appended_member(tmp_path):
    # writer killed mid-member, restarted in the same hour: read_recording raises on this file
    first = gz_member(recording_lines([(1 * S, HB), (2 * S, HB)]))
    second = gz_member(recording_lines([(3 * S, HB)]))
    write_file(tmp_path, "2026-10-18T02", first[:-12] + second)
    write_file(tmp_path, "2026-10-18T03", second)
    recs = list(RecordingReader(tmp_path))
    assert KIND_TRUNCATED in kinds(recs)
    assert kinds(recs)[-2:] == [KIND_META, KIND_MSG]  # next file still read


def test_partial_last_line(tmp_path):
    lines = recording_lines([(1 * S, HB)]) + ['{"recv_ns":2000000000,"msg":{"chan']
    recs = list(RecordingReader(files=[write_file(tmp_path, "2026-10-18T02", gz_member(lines))]))
    assert kinds(recs) == [KIND_META, KIND_MSG, KIND_TRUNCATED]
    assert recs[-1].line_no == 3


def test_symbol_filter_and_time_range(tmp_path):
    btc, eth = {"symbol": "BTC/USD", "checksum": 1}, {"symbol": "ETH/USD", "checksum": 2}
    msgs = [(1 * S, {"channel": "level3", "type": "update", "data": [btc, eth]}),
            (2 * S, {"channel": "level3", "type": "update", "data": [eth]}),
            (3 * S, HB)]
    write_file(tmp_path, "2026-10-18T02", gz_member(recording_lines(msgs)))
    write_file(tmp_path, "2026-10-18T03", gz_member(recording_lines([(3601 * S, HB)])))
    recs = list(RecordingReader(tmp_path, symbols=["BTC/USD"], end=datetime(2026, 10, 18, 3, tzinfo=timezone.utc)))
    assert kinds(recs) == [KIND_META, KIND_MSG, KIND_MSG]  # ETH-only update dropped, 03h file excluded
    assert list(book_items(recs[1])) == [("update", btc)]
    start = datetime(2026, 10, 18, 3, 30, tzinfo=timezone.utc)
    assert [r.file.name for r in RecordingReader(tmp_path, start=start)][0] == "l3_2026-10-18T03.jsonl.gz"


def test_empty_root(tmp_path):
    assert list(RecordingReader(tmp_path)) == []
