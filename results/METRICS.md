# Traffic controller results

**Grid:** 1×1 · **Seeds:** 10 · **Episode:** 1000 steps · **Checkpoint:** `checkpoints/dqn_final.pt`

Agent: Double + dueling DQN with prioritized replay (sum-tree), 2-step returns, soft target updates, and multi-seed best-checkpoint selection.

## Summary (mean ± std across seeds)

| Metric                        | Fixed timer |             DQN |          Δ |
| ----------------------------- | ----------: | --------------: | ---------: |
| Avg stopped / step            | 5.05 ± 0.19 | **3.68 ± 0.24** | **−27.1%** |
| Total wait (car-steps)        |        5050 |        **3682** | **−27.1%** |
| Wait per completed car        |        9.88 |        **7.21** | **−27.1%** |
| Peak congestion (max stopped) |        26.9 |        **25.3** |  **−5.9%** |
| Phase switches / episode      |        49.0 |            50.5 |      +3.1% |
| Throughput (cars completed)   |       511.0 |           511.0 |         0% |

Throughput is identical — the DQN wins by **reducing delay**, not by moving more cars. Phase switching stays near the baseline rate (no light flicker), thanks to the switch penalty in the reward.

## Figures

| File                                                   | Type                   | What it shows                                      |
| ------------------------------------------------------ | ---------------------- | -------------------------------------------------- |
| [comparison_bars.png](./comparison_bars.png)           | Zoomed bar + seed dots | Means with auto-scaled y-axis and per-seed scatter |
| [distribution_boxplot.png](./distribution_boxplot.png) | Box plot               | Spread across 10 seeds — stability / variance      |
| [stopped_timeseries.png](./stopped_timeseries.png)     | Line + band            | Congestion over episode (mean ± std)               |
| [cumulative_wait.png](./cumulative_wait.png)           | Line + band            | Total delay accumulated step-by-step               |
| [improvement_lollipop.png](./improvement_lollipop.png) | Lollipop               | % reduction per metric                             |

Raw numbers: [metrics.json](./metrics.json)

## Reproduce

```bash
source .venv/bin/activate
python scripts/evaluate.py --checkpoint checkpoints/dqn_final.pt --seeds 10
open results/comparison_bars.png
```

## What changed vs the earlier model

| Aspect          | Before                  | After                      |
| --------------- | ----------------------- | -------------------------- |
| Wait reduction  | ~24.7%                  | **27.1%**                  |
| Peak congestion | −3.3%                   | **−5.9%**                  |
| Replay          | uniform                 | **prioritized (sum-tree)** |
| Returns         | 1-step                  | **2-step**                 |
| Checkpoint pick | 1-seed eval             | **8-seed eval**            |
| Switching       | up to ~100/ep (flicker) | **~50/ep (stable)**        |
