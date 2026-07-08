from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from trafficsim.config import SimConfig
from trafficsim.sim import TrafficSim


class TrafficGridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        config: SimConfig | None = None,
        max_steps: int = 1000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.config = config or SimConfig()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.sim = TrafficSim(self.config)
        self.num_intersections = self.config.grid_size ** 2
        obs_dim = 4 + 3 + 1
        flat_dim = obs_dim if self.num_intersections == 1 else obs_dim * self.num_intersections
        high = np.array(
            [self.config.queue_window] * 4 + [1.0] * 4,
            dtype=np.float32,
        )
        if self.num_intersections > 1:
            high = np.tile(high, self.num_intersections)
        self.observation_space = spaces.Box(
            low=0.0,
            high=high,
            shape=(flat_dim,),
            dtype=np.float32,
        )
        self.action_space = (
            spaces.Discrete(2)
            if self.num_intersections == 1
            else spaces.MultiDiscrete([2] * self.num_intersections)
        )
        self._step_count = 0
        self._renderer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.sim.reset(seed=seed)
        else:
            self.sim.reset()
        self._step_count = 0
        obs = self._flat_obs()
        return obs, self._info()

    def step(
        self, action: int | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        actions = self._expand_action(action)
        result = self.sim.step(actions)
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        reward = result.reward
        # Penalize lane starvation so the agent can't ignore one axis forever.
        if self.num_intersections == 1:
            ns_q = result.observations[0, 0] + result.observations[0, 2]
            ew_q = result.observations[0, 1] + result.observations[0, 3]
            reward -= 0.25 * max(ns_q, ew_q)
        reward = float(reward)
        return (
            self._flat_obs(result.observations),
            reward,
            False,
            truncated,
            self._info(result),
        )

    def render(
        self,
        *,
        overlay: str | None = None,
        hud_extra: list[str] | None = None,
    ) -> np.ndarray | None:
        if self.render_mode is None:
            return None
        if self._renderer is None:
            from trafficsim.render import TrafficRenderer

            self._renderer = TrafficRenderer(self.sim.config)
            self._renderer.ensure_ready(self.render_mode)
        return self._renderer.render(
            self.sim,
            mode=self.render_mode,
            overlay=overlay,
            hud_extra=hud_extra,
        )

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _flat_obs(self, observations: np.ndarray | None = None) -> np.ndarray:
        obs = observations if observations is not None else self.sim.get_observations()
        if self.num_intersections == 1:
            return obs[0]
        return obs.reshape(-1)

    def _expand_action(self, action: int | np.ndarray) -> np.ndarray:
        if self.num_intersections == 1:
            return np.array([int(action)], dtype=np.int32)
        if isinstance(action, (int, np.integer)):
            return np.full(self.num_intersections, int(action), dtype=np.int32)
        flat = np.asarray(action, dtype=np.int32).reshape(-1)
        if flat.size != self.num_intersections:
            raise ValueError(
                f"Expected {self.num_intersections} actions, got {flat.size}"
            )
        return flat

    def _info(self, result=None) -> dict[str, Any]:
        metrics = self.sim.metrics
        info = {
            "step": metrics.step,
            "stopped_cars": metrics.stopped_cars,
            "completed": metrics.completed,
            "active_cars": metrics.active_cars,
            "total_wait": metrics.total_wait,
        }
        if result is not None:
            info["per_intersection_stopped"] = result.per_intersection_stopped
            info["phase_switches"] = result.phase_switches
        return info
