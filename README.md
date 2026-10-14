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
