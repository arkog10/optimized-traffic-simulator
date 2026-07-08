#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from trafficsim.config import SimConfig
from trafficsim.env import TrafficGridEnv


def fixed_timer_action(step: int, period: int) -> int:
    cycle = step // period
    return 0 if cycle % 2 == 0 else 1


def run_episode(env: TrafficGridEnv, period: int, seed: int) -> dict[str, float]:
    obs, _ = env.reset(seed=seed)
    total_wait = 0
    completed_start = 0
    steps = 0

    while True:
        action = fixed_timer_action(steps, period)
        obs, reward, _, truncated, info = env.step(action)
        total_wait += info["stopped_cars"]
        steps += 1
        if truncated:
            break

    return {
        "avg_stopped": total_wait / max(steps, 1),
        "throughput": info["completed"] - completed_start,
        "steps": steps,
        "total_wait": total_wait,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-timer traffic-light baseline")
    parser.add_argument("--period", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=1)
    args = parser.parse_args()

    config = SimConfig(grid_size=args.grid_size)
    env = TrafficGridEnv(config=config)

    results = [run_episode(env, args.period, seed=i) for i in range(args.seeds)]
    avg_stopped = np.mean([r["avg_stopped"] for r in results])
    avg_throughput = np.mean([r["throughput"] for r in results])
    print(f"fixed_timer period={args.period}")
    print(f"avg_stopped={avg_stopped:.3f}")
    print(f"avg_throughput={avg_throughput:.1f}")
    env.close()


if __name__ == "__main__":
    main()
