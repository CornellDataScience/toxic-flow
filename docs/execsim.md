# Execution simulator: running it and reading the results

The simulator (`toxicflow/execsim`) replays recorded Kraken L3 order-book data and asks what
would have happened to small passive orders placed on a fixed schedule. Design:
`toxic-flow-queue-technical-design.html`. This page covers how to run an experiment, what the
output files mean, and where the implementation differs from the design.

Two kinds of information are kept apart everywhere:

- **Observed**: what the recording contains. Book events, trade prints, data gaps, and the
  *candidates*: historical events that could have reached one of our orders.
- **Counterfactual**: what we conclude about our order. A fill is always a conclusion drawn from
  candidates under stated assumptions. The recorded market never contained our orders.

## Running an experiment

```
python scripts/run_simulation.py data/raw/l3/2026-10-18/l3_2026-10-18T04.jsonl.gz \
    --trades auto --qty 0.001 --interval-s 10 --horizon-s 10
```

| Option | Meaning | Default |
|---|---|---|
| `l3` (positional) | One or more L3 recordings; replayed in hour order | required |
| `--trades FILE... \| auto` | Public-stream recordings of the same hours. `auto` looks for `public/<date>/public_<date>T<HH>.jsonl.gz` beside each L3 file | none (L3-only) |
| `--evidence auto\|l3_only\|trades` | Which evidence settles fills | `auto`: trades if given |
| `--cross-rule stop\|fill_at_cross\|ignore` | What a crossed order may still count (below) | `stop` |
| `--qty Q` or `SYMBOL=Q ...` | Fixed order size, the same for every strategy | required |
| `--strategies`, `--depth-ticks` | Which strategies; ticks behind the touch for `quote_deeper` | all three; 1 |
| `--interval-s`, `--start`, `--end`, `--sides` | The placement schedule | 10 s over the files' hours, both sides |
| `--horizon-s`, `--latency-ms`, `--warmup-s` | Holding horizon, decision-to-insertion delay, time required after a resync | 10, 0, 0 |
| `--symbols`, `--tick SYMBOL=T` | Symbols; tick-size override | pairs in the first file; `10**-price_precision` |
| `--out`, `--run-id`, `--overwrite` | Where results go | `data/sim/<derived id>` |

Every strategy is evaluated on the same requests, size, horizon and latency, and sees the same
book view. The schedule comes from the clock only and is written before the replay starts.

Without `--trades` the run is **L3-only**: the summary and every row say so, and nothing is
reported as trade-confirmed. Asking for `--evidence trades` without recordings is an error.

Several files are replayed in order. Each file starts a new recording session, so the book is
rebuilt from that file's snapshot and any order live across the boundary is censored, not
counted as unfilled. Checksum failures, resnapshots and receive gaps are handled the same way.

## Output: `data/sim/<run_id>/`

| File | One row per | Notes |
|---|---|---|
| `manifest.json` | the run | Settings, input files with sizes and hashes, code commit and whether the tree was dirty, summary counts |
| `schedule.jsonl` | placement request | Written before the replay |
| `orders.jsonl` | (request, strategy) | Placed, skipped or refused by the data gate |
| `fills.jsonl` | fill **candidate** | Observed events, plus what each policy accepted of them |
| `quality.jsonl` | data-quality event | Snapshots, resnapshots, desyncs, gaps, session restarts, timestamp regressions, end of data |

Quantities and prices are decimal strings; times are integer nanoseconds of exchange time.

**`orders.jsonl`**
- `placed`, `skip_reason`, `skip_stage`: whether an order reached the book. A skip is never an
  unfilled order.
- `request_status` / `eligible`: `ELIGIBLE`, `DATA_GATE` (no usable book at decision time) or
  `COVERAGE` (a data break inside the order's window). Only eligible requests enter the rates.
- `queue_ahead_qty`, `queue_ahead_orders`, `touch_qty`, `level_rank`: queue position at insertion.
- `end_reason`, `censored`: `HORIZON` means canceled at expiry. A censored order's outcome is
  unknown after the censoring time.
- `filled_qty_<policy>`, `status_<policy>`, `first_fill_ts_ns_<policy>`,
  `time_to_first_fill_ns_<policy>`, `complete_ts_ns_<policy>`: the counterfactual outcome under
  each policy, for the evidence mode and crossing rule in `fill_model`.
- `crossed`, `cross_ts_ns`, `cross_price`, `cross_qty`, `cross_ahead_qty`, `n_crossings`,
  `inconsistent_crossing`, `resolved_by_cross_<policy>`: see the next section.
- `checksum_covered`: false if the order's level was ever outside the 10 levels Kraken's
  checksum validates.

**`fills.jsonl`**
- `fill_qty` is the full historical quantity of the event, not what we got.
- `mechanism`: `level` (a reduction behind us at our price), `intercept` (a reduction at the
  touch at a worse price than ours), `cross_arrival` (an opposite order arrived at or through
  our price).
- `evidence` / `evidence_rank`: the label from L3 data alone. `trade_evidence` splits the
  quantity by trade match (`TRADE_CONFIRMED`, `TRADE_PLAUSIBLE`, `TRADE_ABSENT`) when trade
  recordings were used; `trade_ids` are the matched prints.
- `event_ref`, `source_file`, `source_line`: the raw line the event came from.
- `after_cross`, `is_cross_event`, `in_horizon`: where the candidate sits relative to the
  order's crossing and horizon.
- `accepted_qty_<policy>`: what that policy counted. This is the only counterfactual part of
  the row; the rest is the same under every rule.

## Crossed orders and the crossing rule

An order is **crossed** when an opposite-side order comes to rest at or through its price while
it is live (for a buy: an ask at or below our price). A real order at that price would have
traded with the arrival, or the arrival would have been rejected if it was post-only. Either
way the recording from that moment on is not a market that contained our order. Before this was
handled, a later trade at a worse price could be booked as our fill.

The crossing is recorded as a fact (`QueueTracker.cross`). What it means for fills is chosen at
settlement, so one replay can be read under all three rules:

| Rule | After the crossing | Use |
|---|---|---|
| `stop` (reporting default) | Nothing from later items counts, and no crossing arrival counts. An order not complete by then is `resolved_by_cross`: outcome unknown | Headline |
| `fill_at_cross` | Optimistic counts the crossing arrival at its own time and quantity and keeps going. Conservative and balanced stop, as above | Upper bound |
| `ignore` | The order keeps resting and every later candidate counts | Comparison with earlier results only |

Why `stop` is the default: it claims nothing about what happened at the crossing. The data
cannot tell a marketable arrival (we would have been filled) from a post-only one (it would
have been rejected), so neither is assumed. Conservative and balanced give the same numbers
under `stop` and `fill_at_cross`; only optimistic differs.

Things to keep in mind when reading the numbers:
- An order with an unknown outcome is not a non-fill. It is left out of the point estimates
  (`P(f|placed)`), like a censored order, and reported separately (`crossed_unfilled`).
- Crossing is not independent of the outcome: it happens when the price moves through the
  order. Read each fill rate with its bounds. The lower bound counts unknown outcomes as
  non-fills, the upper bound as fills.
- Events in the same item as the crossing still count. A sweep that takes our level and rests
  its remainder produces the fill and the crossing together.
- A crossing while orders are still ahead of us never creates a fill by itself.
- `inconsistent_crossing` marks orders crossed without any reduction having reached them first.
- The library functions (`settle`, `analyze`, `evaluate`) default to `ignore` so existing
  callers are unchanged. `scripts/run_simulation.py` and `scripts/compare_strategies.py` use `stop`.

## Reproducing a run

`manifest.json` holds what is needed: `inputs` (file names, sizes, sha256), `config`, `run`,
`schedule`, and `code.commit` with `code.dirty`. If `dirty` is true the results came from
uncommitted code and cannot be attributed to the commit alone.

Rerunning the same command on the same files and code gives the same run id and byte-identical
`.jsonl` files. `manifest.files` has each file's sha256 to check against. Only the `runtime`
block of the manifest (start time, duration, command line) changes between identical runs.

## Differences from the design document

| Design | Implementation | Why |
|---|---|---|
| §11.1 optional `mids.jsonl` | Not written | Markouts are computed in the run; a mid series is Phase 5 |
| §7.5 rows ordered by (symbol, request id, strategy) | Requests in time order, strategies in run order | Matches the analysis; still deterministic |
| §11.2 one settlement | Settlement depends on evidence mode and crossing rule, named in `fill_model` | The crossing rule is new |
| §6.8 `inconsistent_crossing` as a counter | A per-order flag, plus `n_crossings` | A count of crossings is reported separately |
| §7.4 "trades missing" in `quality.jsonl` | In `manifest.trade_attribution` | Trade-feed health is recorded per reduction, not as events |
| §6.3 row 1c, `increase_policy=keep` | The tracker follows the book: an order ahead that grows goes behind us | `L3Book` was corrected (commit f85989e): keeping the position broke Kraken's checksum. This settles open question OQ-2 |
| §8.8 `PlacementSchedule.random_uniform` | Not built | The regular schedule needs no seed |

## Limitations that remain

- Other participants are assumed not to react to our order, and our fills remove no liquidity.
- Under `stop`, many orders can end with an unknown outcome when the spread is one tick, which
  widens the bounds. That is the honest state of the evidence, not a defect to tune away.
- No confidence intervals yet. Orders close in time are correlated.
- Fees and rebates are not modeled.
