# Toxic Flow: implementation group starter code

Fill simulator, markouts, and a synthetic data generator so we can build and
test everything before the data extraction group's real feed is ready.

## Setup
```
pip install -r requirements.txt
python -m pytest            # 14 unit tests, all should pass
python -m examples.run_synthetic   # end-to-end demo, ~3 seconds
```

## Layout
| File | What it does |
|---|---|
| `sim/events.py` | **Data contract** with the extraction group: `BookEvent` (add/modify/delete one order) and `Trade`. Their output gets converted to these. |
| `sim/book.py` | L3 order book; each price level is a FIFO queue of orders |
| `sim/fill_sim.py` | `QueueAwareSimulator` (must wait for the queue ahead) and `NaiveSimulator` (fills on first touch) |
| `sim/backtest.py` | Replays events through the book + both simulators; `JoinBestQuoter` is a simple maker that quotes at the best bid/ask |
| `sim/synthetic.py` | Fake L3 market with informed traders, so adverse selection actually exists in the data |
| `analysis/markouts.py` | Markouts at 100ms / 1s / 10s, markout curves, placeholder toxicity label |
| `tests/` | Hand-built book scenarios (OKR 2: "unit-tested against hand-constructed book scenarios") |

## Definitions (agree on these as a team)
- **Markout** = `sign × (mid(t+h) − fill_price)`, sign +1 for a buy, −1 for a sell.
  Positive is good for us. A fill with no price move scores +half spread.
- **Queue position**: our hypothetical order joins the back of the queue. Orders
  ahead of us leave via cancels or trades; new orders join behind us.

## Assumptions to verify on real data
1. **Trade arrives before the book update** for the same match. If reversed,
   the queue-aware simulator can produce phantom fills (see `fill_sim.py`).
2. **Size increases lose priority; decreases keep it.** Check the exchange's rules.
3. **Our orders don't affect the market** (no one reacts to them). State this
   as a limitation in the report.

## Demo output (synthetic data, seed 42)
| | Queue-aware | Naive |
|---|---|---|
| Fills | 3,186 | 15,862 |
| Mean markout @10s (ticks) | +0.16 | +0.53 |

The naive model fills ~5x more often and overstates fill quality by about
3x at the 10s horizon. That gap is the "standard modeling error" from OKR 2,
already visible on fake data. (Synthetic numbers prove the code works, not
anything about real markets.)

## Next tasks
- [ ] Feature engineering (`analysis/features.py`): order flow imbalance, queue
      depletion rate, depth asymmetry, trade-sign autocorrelation, realized vol,
      cancel-to-trade ratio. Compute each from book state *before* the fill.
- [ ] Agree on the toxicity label (horizon + threshold) in `label_toxic`
- [ ] Walk-forward split + gradient-boosted classifier vs spread+vol baseline
- [ ] Adapter from the extraction group's files to `BookEvent`/`Trade`
- [ ] Speed: `best_bid()` scans all levels; switch to `sortedcontainers` once
      real data volumes show it's a bottleneck
