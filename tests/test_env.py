import numpy as np
import pytest

from trafficsim.config import SimConfig
from trafficsim.env import TrafficGridEnv
from trafficsim.sim import TrafficSim


def test_sim_spawns_and_moves():
    sim = TrafficSim(SimConfig(spawn_rate=1.0), seed=0)
    sim.reset(seed=0)
    for _ in range(30):
        sim.step(np.array([0]))
    assert sim.metrics.active_cars >= 0
    assert sim.metrics.step == 30


def test_env_spaces_and_step():
    env = TrafficGridEnv(config=SimConfig(), max_steps=50)
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert terminated is False
        assert isinstance(truncated, bool)
        assert "stopped_cars" in info

    env.close()


def test_env_truncates_at_horizon():
    env = TrafficGridEnv(config=SimConfig(), max_steps=5)
    env.reset(seed=2)
    truncated = False
    for _ in range(10):
        _, _, _, truncated, _ = env.step(0)
        if truncated:
            break
    assert truncated is True
    env.close()


def test_multi_intersection_obs_shape():
    env = TrafficGridEnv(config=SimConfig(grid_size=2), max_steps=20)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (8 * 4,)
    env.close()
