from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        dueling: bool = True,
    ):
        super().__init__()
        self.dueling = dueling
        layers: list[nn.Module] = []
        prev = obs_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        self.feature = nn.Sequential(*layers)

        if dueling:
            self.value_head = nn.Linear(prev, 1)
            self.advantage_head = nn.Linear(prev, action_dim)
        else:
            self.head = nn.Linear(prev, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature(x)
        if not self.dueling:
            return self.head(z)
        value = self.value_head(z)
        advantage = self.advantage_head(z)
        # Subtract mean advantage for identifiability of V vs A.
        return value + advantage - advantage.mean(dim=1, keepdim=True)


@dataclass
class DQNAgent:
    obs_dim: int
    action_dim: int = 2
    hidden_dims: tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    target_sync: int = 500
    double_dqn: bool = True
    dueling: bool = True
    tau: float = 0.01
    grad_clip: float = 10.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.policy_net = QNetwork(
            self.obs_dim, self.action_dim, self.hidden_dims, self.dueling
        ).to(self.device)
        self.target_net = QNetwork(
            self.obs_dim, self.action_dim, self.hidden_dims, self.dueling
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        # Elementwise so we can apply per-sample importance weights (PER).
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.epsilon = 1.0
        self.train_steps = 0

    def act(self, obs: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return int(np.random.randint(self.action_dim))
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        with torch.no_grad():
            q_values = self.policy_net(obs_tensor)
        return int(q_values.argmax(dim=1).item())

    def learn(self, batch: dict) -> tuple[float, np.ndarray]:
        states_t = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(batch["actions"], dtype=torch.int64, device=self.device)
        rewards_t = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
        next_states_t = torch.as_tensor(
            batch["next_states"], dtype=torch.float32, device=self.device
        )
        dones_t = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)
        # Per-transition bootstrap discount (gamma ** k) from n-step folding.
        disc_t = torch.as_tensor(batch["discounts"], dtype=torch.float32, device=self.device)
        weights_t = torch.as_tensor(batch["weights"], dtype=torch.float32, device=self.device)

        q_values = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.policy_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1).values
            target = rewards_t + disc_t * next_q * (1.0 - dones_t)

        td_errors = target - q_values
        loss = (weights_t * self.loss_fn(q_values, target)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.train_steps += 1
        if self.tau > 0:
            self._soft_update()
        elif self.train_steps % self.target_sync == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item()), td_errors.detach().cpu().numpy()

    def _soft_update(self) -> None:
        with torch.no_grad():
            for target_param, param in zip(
                self.target_net.parameters(),
                self.policy_net.parameters(),
                strict=True,
            ):
                target_param.mul_(1.0 - self.tau).add_(param, alpha=self.tau)

    def decay_epsilon(self, epsilon_end: float, epsilon_decay: float) -> None:
        self.epsilon = max(epsilon_end, self.epsilon * epsilon_decay)

    def save(self, path: str) -> None:
        torch.save(
            {
                "policy_state_dict": self.policy_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "train_steps": self.train_steps,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "hidden_dims": self.hidden_dims,
                "dueling": self.dueling,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> DQNAgent:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        agent = cls(
            obs_dim=int(checkpoint["obs_dim"]),
            action_dim=int(checkpoint["action_dim"]),
            hidden_dims=tuple(checkpoint["hidden_dims"]),
            dueling=bool(checkpoint.get("dueling", True)),
            device=device,
        )
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        agent.epsilon = float(checkpoint.get("epsilon", 0.0))
        agent.train_steps = int(checkpoint.get("train_steps", 0))
        agent.policy_net.eval()
        return agent
