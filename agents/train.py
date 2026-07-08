from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.dqn import DQNAgent
from agents.replay_buffer import (
    NStepAccumulator,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
)
from trafficsim.config import SimConfig, TrainConfig
from trafficsim.env import TrafficGridEnv


def _local_obs_dim(env: TrafficGridEnv) -> int:
    dim = int(np.prod(env.observation_space.shape))
    num_ix = env.config.grid_size**2
    return dim if num_ix == 1 else dim // num_ix


def greedy_eval(
    env: TrafficGridEnv,
    agent: DQNAgent,
    obs_dim: int,
    seeds: list[int],
) -> float:
    """Average greedy avg-stopped over several seeds (lower is better)."""
    num_ix = env.config.grid_size**2
    scores = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        stopped_total = 0
        steps = 0
        while True:
            if num_ix == 1:
                action = agent.act(obs, explore=False)
            else:
                local = obs.reshape(num_ix, obs_dim)
                action = np.array(
                    [agent.act(local[i], explore=False) for i in range(num_ix)],
                    dtype=np.int32,
                )
            obs, _, _, truncated, info = env.step(action)
            stopped_total += info["stopped_cars"]
            steps += 1
            if truncated:
                break
        scores.append(stopped_total / max(steps, 1))
    return float(np.mean(scores))


def train(
    sim_config: SimConfig | None = None,
    train_config: TrainConfig | None = None,
    seed: int = 42,
) -> DQNAgent:
    sim_config = sim_config or SimConfig()
    train_config = train_config or TrainConfig()
    env = TrafficGridEnv(config=sim_config, max_steps=train_config.max_steps)
    obs_dim = _local_obs_dim(env)
    num_ix = sim_config.grid_size**2

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=2,
        hidden_dims=train_config.hidden_dims,
        gamma=train_config.gamma,
        lr=train_config.lr,
        batch_size=train_config.batch_size,
        target_sync=train_config.target_sync,
        double_dqn=train_config.double_dqn,
        dueling=train_config.dueling,
        tau=train_config.tau,
        grad_clip=train_config.grad_clip,
    )

    if train_config.prioritized:
        buffer: ReplayBuffer = PrioritizedReplayBuffer(
            capacity=train_config.buffer_size,
            obs_dim=obs_dim,
            batch_size=train_config.batch_size,
            alpha=train_config.per_alpha,
        )
    else:
        buffer = ReplayBuffer(
            capacity=train_config.buffer_size,
            obs_dim=obs_dim,
            batch_size=train_config.batch_size,
        )

    nstep = NStepAccumulator(train_config.n_step, train_config.gamma)

    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    scale = train_config.reward_scale
    total_steps = train_config.episodes * train_config.max_steps
    beta0 = train_config.per_beta_start
    eval_seeds = [10_000 + i for i in range(train_config.eval_seeds)]
    env_steps = 0
    best_avg_stopped = float("inf")

    for episode in range(1, train_config.episodes + 1):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        nstep.reset()
        stopped_total = 0
        info: dict = {"step": 0, "completed": 0, "stopped_cars": 0}

        for _ in range(train_config.max_steps):
            if num_ix == 1:
                action = agent.act(obs)
                next_obs, reward, _, truncated, info = env.step(action)
                for t in nstep.push(0, obs, action, reward * scale, next_obs, truncated):
                    buffer.push(t)
            else:
                local_obs = obs.reshape(num_ix, obs_dim)
                actions = np.array(
                    [agent.act(local_obs[i]) for i in range(num_ix)],
                    dtype=np.int32,
                )
                next_obs, _, _, truncated, info = env.step(actions)
                next_local = next_obs.reshape(num_ix, obs_dim)
                for i in range(num_ix):
                    local_reward = -float(info["per_intersection_stopped"].get(i, 0))
                    for t in nstep.push(
                        i,
                        local_obs[i],
                        int(actions[i]),
                        local_reward * scale,
                        next_local[i],
                        truncated,
                    ):
                        buffer.push(t)

            env_steps += 1
            if env_steps >= train_config.learning_starts:
                beta = min(1.0, beta0 + (1.0 - beta0) * env_steps / total_steps)
                for _ in range(train_config.updates_per_step):
                    batch = buffer.sample(beta=beta)
                    if batch is None:
                        break
                    _loss, td_errors = agent.learn(batch)
                    buffer.update_priorities(batch["indices"], td_errors)

            obs = next_obs
            stopped_total += info["stopped_cars"]
            if truncated:
                break

        agent.decay_epsilon(train_config.epsilon_end, train_config.epsilon_decay)
        train_avg = stopped_total / max(info["step"], 1)

        if episode % train_config.log_every == 0 or episode == 1:
            eval_avg = greedy_eval(env, agent, obs_dim, eval_seeds)
            if eval_avg < best_avg_stopped:
                best_avg_stopped = eval_avg
                agent.save(str(checkpoint_dir / "dqn_best.pt"))
            print(
                f"episode={episode:4d} train_avg={train_avg:5.2f} "
                f"eval_avg={eval_avg:5.2f} (best {best_avg_stopped:5.2f}) "
                f"eps={agent.epsilon:.3f} completed={info['completed']}"
            )

    final_path = checkpoint_dir / "dqn_final.pt"
    agent.save(str(final_path))
    best_path = checkpoint_dir / "dqn_best.pt"
    if best_path.exists():
        DQNAgent.load(str(best_path)).save(str(final_path))
        print(f"promoted {best_path} (avg_stopped={best_avg_stopped:.2f}) to {final_path}")
    else:
        print(f"saved checkpoint to {final_path}")
    env.close()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN traffic-light controller")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--grid-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sim_config = SimConfig(grid_size=args.grid_size)
    train_config = TrainConfig(episodes=args.episodes)
    train(sim_config=sim_config, train_config=train_config, seed=args.seed)


if __name__ == "__main__":
    main()
