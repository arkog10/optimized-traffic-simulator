#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pygame

from trafficsim.config import SimConfig
from trafficsim.render import TrafficRenderer
from trafficsim.sim import TrafficSim
from trafficsim.viewer import SpeedController

END_OVERLAY = "Simulation ended\nR = restart   Q = quit"


def fixed_timer_action(step: int, period: int) -> int:
    cycle = step // period
    return 0 if cycle % 2 == 0 else 1


def reset_sim(config: SimConfig, seed: int) -> tuple[TrafficSim, int]:
    sim = TrafficSim(config, seed=seed)
    sim.reset(seed=seed)
    return sim, 0


def hud_lines(paused: bool, speed: SpeedController) -> list[str]:
    state = "paused" if paused else "running"
    return [
        f"Speed     {speed.label}",
        f"Status    {state}",
        "Space pause  +/- speed",
        "Right step   R restart",
        "Q quit",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual smoke test with fixed-timer lights")
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Stop after N sim steps (0 = run until you quit)",
    )
    parser.add_argument("--grid-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    config = SimConfig(grid_size=args.grid_size)
    seed = args.seed
    sim, step = reset_sim(config, seed)

    renderer = TrafficRenderer(config)
    renderer.ensure_ready()
    speed = SpeedController(preset_index=1)
    paused = False
    finished = False
    running = True

    renderer.render(
        sim,
        hud_extra=hud_lines(paused, speed),
    )

    while running:
        dt = min(renderer.tick_dt(), 0.1)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                finished = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    sim, step = reset_sim(config, seed)
                    speed.reset()
                    paused = False
                    finished = False
                elif event.key == pygame.K_RIGHT and not finished:
                    action = fixed_timer_action(step, 20)
                    actions = np.full(config.grid_size**2, action, dtype=np.int32)
                    sim.step(actions)
                    step += 1
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed.faster()
                elif event.key == pygame.K_MINUS:
                    speed.slower()

        if finished:
            renderer.render(
                sim,
                overlay=END_OVERLAY,
                hud_extra=hud_lines(paused, speed),
            )
            continue

        if not paused:
            for _ in range(speed.tick(dt)):
                action = fixed_timer_action(step, 20)
                actions = np.full(config.grid_size**2, action, dtype=np.int32)
                sim.step(actions)
                step += 1
                if args.steps > 0 and step >= args.steps:
                    finished = True
                    break

        renderer.render(
            sim,
            hud_extra=hud_lines(paused, speed),
        )

    renderer.close()


if __name__ == "__main__":
    main()
