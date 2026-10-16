"""End-to-end demo on synthetic data: simulate fills, compare naive vs queue-aware.

Run from the repo root:  python -m examples.run_synthetic
"""
from analysis.markouts import compute_markouts, markout_curve
from sim.backtest import JoinBestQuoter, run_backtest
from sim.synthetic import generate


def main():
    events = generate(n_events=200_000, seed=42)
    fills, mids = run_backtest(events, JoinBestQuoter(size=1.0, min_interval=0.5))
    print(f"{len(events):,} events, {len(mids):,} mid changes\n")
    for name, f in fills.items():
        m = compute_markouts(f, mids)
        print(f"== {name}: {len(f):,} fills, mean time to fill {f['time_to_fill'].mean():.2f}s")
        print(markout_curve(m).to_string(index=False), "\n")


if __name__ == "__main__":
    main()
