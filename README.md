# RL Traffic Grid

Cellular-automata traffic sim with a hand-written DQN controlling traffic lights to minimize wait time.

On a single intersection the trained agent cuts average wait **~27%** vs a fixed timer (see [results/METRICS.md](results/METRICS.md)).

## DQN features

Hand-rolled in PyTorch (no RL library):

- Double DQN + dueling network heads
- Prioritized experience replay (sum-tree, O(log N) sampling)
- N-step returns for faster reward propagation
- Soft target updates (Polyak) + gradient clipping
- Best-checkpoint selection via multi-seed greedy eval

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use `pip install -e .` so the `trafficsim` and `agents` packages import correctly.

## Quick start

```bash
# 1. Sanity: unit tests
pytest -q

# 2. Baseline visual (fixed-timer lights, no ML)
python scripts/random_drive.py

# 3. Train the DQN on one intersection (~13 min CPU)
python -m agents.train --episodes 500 --grid-size 1

# 4. Compare trained vs baseline (writes plots + metrics.json to results/)
python scripts/evaluate.py --checkpoint checkpoints/dqn_final.pt --seeds 10

# 5. Watch the trained agent drive the lights
python scripts/play.py --checkpoint checkpoints/dqn_final.pt
```

A trained checkpoint ships in `checkpoints/dqn_final.pt`, so you can skip step 3 and jump straight to evaluate/play.

Fixed-timer metrics only (no checkpoint needed):

```bash
python scripts/baseline.py --seeds 10
```

Train / watch a 2x2 grid (shared-parameter DQN):

```bash
python -m agents.train --episodes 600 --grid-size 2
python scripts/play.py --checkpoint checkpoints/dqn_final.pt --grid-size 2
```

## Controls (play / random_drive)

| Key         | Action                     |
| ----------- | -------------------------- |
| Space       | Pause / resume             |
| `+` / `-`   | Speed (2–12 steps/sec)     |
| Right arrow | Single step (random_drive) |
| `r`         | Restart                    |
| `q` / Esc   | Quit                       |

Closing the window shows an end screen — press **R** to restart or **Q** to quit. Clicking away from the window no longer stops the sim.

## Architecture

- `trafficsim/` — numpy sim core, gymnasium env, pygame renderer
- `agents/` — PyTorch DQN, replay buffer, training loop
- `scripts/` — baseline, evaluate, play, smoke test

Sim and RL are decoupled: training runs headless; rendering only reads sim state.

## Metrics

`scripts/evaluate.py` reports these (mean over N seeds) and saves 5 figures + `metrics.json` to `results/`:

- **avg_stopped** — mean stopped cars per step (lower is better)
- **total_wait** — cumulative stopped-car steps
- **wait_per_car** — total wait / cars completed
- **max_stopped** — peak congestion in any single step
- **phase_switches** — light changes per episode (flicker check)
- **throughput** — cars that exited the grid

See [results/METRICS.md](results/METRICS.md) for the latest numbers and chart descriptions.

## Tests

```bash
pytest -q
```
