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

from agents.dqn import DQNAgent
from trafficsim.config import SimConfig
from trafficsim.env import TrafficGridEnv
from trafficsim.grid import Phase
from trafficsim.viewer import SpeedController

END_OVERLAY = "Episode ended\nR = restart   Q = quit"


def phase_label(phase: Phase) -> str:
    if phase == Phase.NS_GREEN:
        return "NS green"
    if phase == Phase.EW_GREEN:
        return "EW green"
    return "yellow / all red"


def hud_lines(paused: bool, speed: SpeedController, env: TrafficGridEnv) -> list[str]:
    state = "paused" if paused else "running"
    lights = phase_label(env.sim.intersections[0].phase)
    return [
        f"Mode      DQN (trained)",
        f"Lights    {lights}",
        f"Speed     {speed.label}",
        f"Status    {state}",
        "Space pause  +/- speed",
        "R restart    Q quit",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Play trained DQN with Pygame UI")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dqn_final.pt")
    parser.add_argument("--grid-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = SimConfig(grid_size=args.grid_size)
    env = TrafficGridEnv(config=config, render_mode="human")
    agent = DQNAgent.load(args.checkpoint)
    agent.epsilon = 0.0

    num_ix = config.grid_size**2
    local_dim = int(np.prod(env.observation_space.shape))
    if num_ix > 1:
        local_dim = local_dim // num_ix

    seed = args.seed
    speed = SpeedController(preset_index=1)
    paused = False
    finished = False
    running = True

    def reset_episode() -> np.ndarray:
        obs, _ = env.reset(seed=seed)
        speed.reset()
        return obs

    obs = reset_episode()
    env.render(hud_extra=hud_lines(paused, speed, env))
    renderer = env._renderer
    assert renderer is not None

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
                    obs = reset_episode()
                    paused = False
                    finished = False
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed.faster()
                elif event.key == pygame.K_MINUS:
                    speed.slower()

        if finished:
            env.render(
                overlay=END_OVERLAY,
                hud_extra=hud_lines(paused, speed, env),
            )
            continue

        if not paused:
            for _ in range(speed.tick(dt)):
                if num_ix == 1:
                    action = agent.act(obs, explore=False)
                else:
                    local_obs = obs.reshape(num_ix, local_dim)
                    action = np.array(
                        [agent.act(local_obs[i], explore=False) for i in range(num_ix)],
                        dtype=np.int32,
                    )
                obs, _, _, truncated, _ = env.step(action)
                if truncated:
                    finished = True
                    break

        env.render(hud_extra=hud_lines(paused, speed, env))

    env.close()


if __name__ == "__main__":
    main()
