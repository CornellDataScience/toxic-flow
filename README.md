# toxic-flow

Measuring and predicting adverse selection on passive fills, built on order
book data reconstructed from Kraken.

This repo currently contains the **data layer**: Kraken WebSocket v2
collectors for L2 (`book`), L3 (`level3`, order by order), and trades. It
maintains local books validated against Kraken's CRC32 checksums and
archives the raw messages for replay.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then paste the Kraken key + secret into .env
pytest                      # offline unit tests (checksums, signing, book logic)
```

Scripts can be run directly (`python scripts/<name>.py`). They add the repo
root to `sys.path`, so they don't depend on the editable install.

### API key

Only the **L3 feed needs a key**. L2 and trades are public. Create the key at
Kraken → Settings → API with **only** these permissions:

- **WebSocket interface**: required (the L3 feed authenticates with a token from `GetWebSocketsToken`)
- Query Funds: optional (`test_api_key.py` uses it as a signature check)

Leave every trading, deposit, and withdrawal permission off. The project only
reads data.

### Secrets

`.env` holds the key and is gitignored. Everything else in the repo is safe to
commit and share. Each person who runs the L3 collector needs their own `.env`.
Don't share keys over git, Slack, or notebooks. Use a password manager, or
better, have each person generate their own read-only key. Kraken nonces must
increase per key, so two people using the same key at the same moment can hit
`EAPI:Invalid nonce`.

## Scripts

| script | key? | what it does |
|---|---|---|
| `scripts/test_api_key.py` | yes | Checks the key end to end: REST signature, WS token, L3 subscribe, and the L3 snapshot checksum |
| `scripts/stream_l2_book.py` | no | Live L2 book + trades; prints top of book and checks every checksum |
| `scripts/stream_l3_orders.py` | yes | Live L3 add/modify/delete events and a periodic book summary (`--events` for every event) |
| `scripts/record.py` | L3 only | Collector: writes raw L2 + trades (+ L3) to `data/raw/`, rotating files hourly |
| `scripts/validate_recording.py` | no | Replays recorded files offline; reports checksum failure rate, message counts, gaps/uptime |

Typical first run:

```bash
python scripts/stream_l2_book.py --symbol BTC/USD --seconds 30 --trades   # no key needed
python scripts/test_api_key.py                                            # once .env is filled in
python scripts/stream_l3_orders.py --symbol BTC/USD --seconds 30 --events
python scripts/record.py --symbol BTC/USD --symbol ETH/USD --seconds 600
python scripts/validate_recording.py data/raw/public/*/*.jsonl.gz
python scripts/validate_recording.py data/raw/l3/*/*.jsonl.gz
```

Symbols use Kraken v2 names (`BTC/USD`, not `XBT/USD`).

## How it works

- **`toxicflow/kraken/book.py`**: `L2Book` (price → qty) and `L3Book` (price →
  individual orders). L3 orders within a level stay in Kraken's order,
  because the L3 checksum hashes them in that order.
- **Checksums**: after every message, the top 10 levels are CRC32-hashed and
  compared with Kraken's value. Prices are parsed as `Decimal` and formatted
  with the pair's `price_precision`/`qty_precision`. Kraken sends JSON
  numbers, which drop trailing zeros, so this precision is needed. On a
  mismatch the symbol is unsubscribed and resubscribed to get a fresh
  snapshot.
- **`toxicflow/kraken/ws.py`**: `BookStream` handles connect, subscribe,
  resync on mismatch, reconnect with backoff, and a stale-connection watchdog
  (30 s without messages; Kraken heartbeats about once a second). L3 gets a
  new auth token on every reconnect.
- **`toxicflow/kraken/recorder.py`**: one gzipped JSONL file per stream per UTC
  hour at `data/raw/<stream>/<date>/<stream>_<date>T<HH>.jsonl.gz`. The first
  line is metadata (depth, precisions). After that, each line is
  `{"recv_ns": <local receive time>, "msg": <Kraken's exact JSON>}`. Every
  file begins with a fresh snapshot, so each hour can be replayed on its own.

## Book query library (`toxicflow/bookquery`)

Answers "what did the book look like at time t?" for the fill simulator,
markouts and ML features.

```python
import glob
from toxicflow.bookquery import BookQuery, build_tables

# Until the event log / valid_intervals tables exist, build equivalents from raw files.
tables = build_tables(glob.glob("data/raw/l3/*/*.jsonl.gz"),
                      glob.glob("data/raw/public/*/*.jsonl.gz"), symbol="BTC/USD")
bq = BookQuery.load(tables.events, tables.trades, tables.valid_intervals,
                    pair=tables.pair, depth=tables.depth,
                    clock="exchange", cache_dir="data/cache/bookquery")

t = "2026-10-16T21:08:00Z"                  # or int ns, datetime, pd.Timestamp
bq.best_at(t)                               # Best(bid_price, bid_qty, ask_price, ask_qty) as Decimal
bq.mid_at(t), bq.spread_at(t)               # float; also take arrays of times (vectorized)
bq.depth_at(t, levels=10)                   # {"bids": [(price, size), ...], "asks": [...]}
bq.queue_at("bid", "81537.4", t)            # [(order_id, size), ...], front of queue first
bq.queue_ahead("bid", "81537.4", t)         # size a new order joining now would wait behind
bq.queue_ahead("bid", "81537.4", t, order_id="OTPONL-XSSW7-WQVKCF")
bq.mid_after(fill_times, ("100ms", "1s", "10s"))   # DataFrame for markouts
bq.is_valid(t)
for step in bq.iter_book(t0, t1):           # sequential scan for the fill simulator
    step.ts, step.book, step.events, step.reset
```

`scripts/markouts_example.py` computes markouts for a list of fills (recorded
trades by default). `scripts/bench_bookquery.py` reproduces the timings below.

### Rules

- **As-of:** a query at `t` reflects every event with `ts <= t`, in original
  message order (`seq`), and nothing after. All events at one timestamp
  apply together. No lookahead, so features are safe for walk-forward evaluation.
- **Clock:** times are int ns since the epoch (UTC). `clock="exchange"`
  (default) uses Kraken's per-message `timestamp`. `clock="recv"` uses
  local receive time. Horizons are in the same clock. Trade timestamps are
  in the same exchange clock as L3 but truncated to microseconds, so a
  fill's own L3 events land a few hundred ns *after* the trade's
  timestamp. `mid_at(trade_ts)` is therefore the pre-trade mid.
- **Validity:** outside a valid interval, queries return `None`/`NaN`, or
  raise `InvalidTimeError` with `strict=True`. Every interval starts from a
  fresh snapshot, so no book is carried across a gap. `mid_after(t, h)` is
  `NaN` if `t + h` leaves the interval containing `t`. The adapter ends an
  interval at a checksum mismatch (1 ns before the bad message), a silence
  longer than `gap_s`, a recorder restart, a new snapshot, or the end of data.
- **Queue priority** (each level is a FIFO queue, so position decides who
  fills first): adds join the back of the queue. A size *decrease* (partial
  fill or amend down) keeps the order's place. A size *increase* sends it to
  the back, since exchanges won't let you jump the queue by enlarging an
  order. Price changes arrive as delete + add. Only a delete removes an
  order. These rules reproduce Kraken's checksum on every recorded message.
- **Depth caveat:** the feed shows only the top `depth` levels per side
  (100 in our recordings). Levels pushed past that are dropped, and anything
  beyond the deepest visible level is *unknown, not empty*. `queue_at` and
  `queue_ahead` there emit `OutsideDepthWarning`. Some `add` events are
  "refills": old orders scrolling into view from below level 100, not new
  orders. Their own `timestamp` field is older than the message.
- **Precision:** prices and sizes are integer ticks/lots internally, and
  book results are `Decimal`. `mid`/`spread`/array `best_at` are float64,
  converted only through `units.ticks_to_float`.

### Input tables (for swapping in the event log / valid_intervals outputs)

| table | required columns | optional |
|---|---|---|
| events | `ts_<clock>` or `ts` (int ns), `seq` (message order), `order_id`, `side` (bid/ask), `event` (snapshot/add/modify/delete), `price_ticks`+`qty_lots` **or** `price`+`qty` (remaining size after the event) | `msg_id` (depth truncation per message), `symbol`, any extra columns (e.g. classification) pass through in `bq.events` |
| trades | `ts_<clock>` or `ts`, price, qty, `side` (aggressor) | anything |
| valid_intervals | `start_<clock>`/`end_<clock>` or `start`/`end` (inclusive ns) | `symbol` |

Each interval's events must begin with the book's snapshot, because replay
starts from an empty book at `start`. If the event log drops snapshot
orders or `seq`, the library can't rebuild the book.

### Benchmarks

Measured on the 2.5-minute BTC/USD sample (14k events, 9.8k messages),
Apple Silicon, Python 3.12, `checkpoint_every=500`:

| operation | time |
|---|---|
| `build_tables` (raw replay + checksum per message) | 340 ms |
| `BookQuery.load` (build / from disk cache) | 53 ms / 15 ms |
| 10k random `mid_at`, one array call | 0.4 ms (0.04 µs each) |
| 10k random `mid_at`, scalar loop | 180 ms (18 µs each) |
| 10k `mid_after` × 3 horizons | 1.8 ms |
| 1k random `book_at` | 500 ms (0.5 ms each) |
| `iter_book` over all 8k timestamps | 51 ms |

Pass arrays to the top-of-book functions: a single call is ~400× faster
than looping. `book_at` costs ~4 µs × events replayed since the last
checkpoint, so smaller `checkpoint_every` trades memory for speed. For
walking forward in time, `iter_book` is far cheaper than repeated `book_at`.

## Troubleshooting

- `CERTIFICATE_VERIFY_FAILED`: the code already uses `certifi`'s CA bundle. If
  you still see this from other tools, run `Install Certificates.command`
  from your python.org Python folder.
- `ModuleNotFoundError: toxicflow` with `pip install -e` on macOS: if the repo
  is in an iCloud-synced folder (Desktop/Documents), iCloud marks `.venv`
  files hidden and Python 3.13+ skips hidden `.pth` files. The scripts work
  around this. Moving the repo out of iCloud is better.
- `ESession:Invalid session` on the L3 subscribe means the token was rejected.
  Run `test_api_key.py`.
