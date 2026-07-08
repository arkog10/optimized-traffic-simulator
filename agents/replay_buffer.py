from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    discount: float  # gamma ** k for the bootstrap term (k = steps folded in)


class NStepAccumulator:
    """Folds up to n_step rewards into a single transition per stream.

    Each intersection is its own stream (keyed by int) so multi-agent
    training keeps trajectories separate. Emits ready n-step transitions
    as they mature and flushes partial ones at episode end.
    """

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self._streams: dict[int, deque] = {}

    def _stream(self, key: int) -> deque:
        if key not in self._streams:
            self._streams[key] = deque(maxlen=self.n_step)
        return self._streams[key]

    def push(
        self,
        key: int,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> list[Transition]:
        buf = self._stream(key)
        buf.append((state, action, reward, next_state, done))
        out: list[Transition] = []
        warmed = len(buf) == self.n_step
        if warmed:
            out.append(self._fold(list(buf)))  # full n-step anchored at buf[0]
        if done:
            # Flush remaining (shorter) windows so tail states still learn.
            start = 1 if warmed else 0
            tail = list(buf)[start:]
            while tail:
                out.append(self._fold(tail))
                tail = tail[1:]
            buf.clear()
        return out

    def _fold(self, window: list) -> Transition:
        state, action = window[0][0], window[0][1]
        reward = 0.0
        discount = 1.0
        next_state = window[-1][3]
        done = False
        for _s, _a, r, ns, d in window:
            reward += discount * r
            discount *= self.gamma
            next_state = ns
            if d:
                done = True
                break
        return Transition(state, action, reward, next_state, done, discount)

    def reset(self) -> None:
        self._streams.clear()


class ReplayBuffer:
    """Uniform experience replay with n-step discounts."""

    def __init__(self, capacity: int, obs_dim: int, batch_size: int = 64):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.batch_size = batch_size
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.discounts = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, t: Transition) -> None:
        idx = self.position
        self.states[idx] = t.state
        self.actions[idx] = t.action
        self.rewards[idx] = t.reward
        self.next_states[idx] = t.next_state
        self.dones[idx] = float(t.done)
        self.discounts[idx] = t.discount
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, beta: float = 0.0) -> dict | None:
        if self.size < self.batch_size:
            return None
        idx = np.random.randint(0, self.size, size=self.batch_size)
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
            "discounts": self.discounts[idx],
            "weights": np.ones(self.batch_size, dtype=np.float32),
            "indices": idx,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        return  # no-op for uniform buffer

    def __len__(self) -> int:
        return self.size


class _SumTree:
    """Fixed-capacity binary tree whose leaves store priorities and whose
    internal nodes store subtree sums. Sampling and updates are O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)

    def total(self) -> float:
        return float(self.tree[1])

    def set(self, leaf: int, value: float) -> None:
        i = leaf + self.capacity
        self.tree[i] = value
        i >>= 1
        while i >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i >>= 1

    def get(self, leaf: int) -> float:
        return float(self.tree[leaf + self.capacity])

    def set_batch(self, leaves: np.ndarray, values: np.ndarray) -> None:
        i = leaves + self.capacity
        self.tree[i] = values
        i = np.unique(i >> 1)
        while True:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            if i[0] == 1:
                break
            i = np.unique(i >> 1)

    def find_batch(self, prefixes: np.ndarray) -> np.ndarray:
        """Vectorized descent: leaf indices whose ranges contain `prefixes`.

        All root->leaf paths have equal length (leaves occupy a contiguous
        block), so a single fixed-depth loop over numpy arrays suffices.
        """
        idx = np.ones(len(prefixes), dtype=np.int64)
        prefixes = prefixes.copy()
        while idx[0] < self.capacity:
            left = 2 * idx
            go_left = prefixes <= self.tree[left]
            prefixes = np.where(go_left, prefixes, prefixes - self.tree[left])
            idx = np.where(go_left, left, left + 1)
        return idx - self.capacity


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional prioritized replay (Schaul et al. 2016) with n-step.

    Samples transitions in proportion to |TD error|^alpha and corrects the
    resulting bias with importance-sampling weights annealed via beta.
    A sum-tree keeps sampling/updates at O(log N) so large buffers stay fast.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        batch_size: int = 64,
        alpha: float = 0.6,
        eps: float = 1e-5,
    ):
        super().__init__(capacity, obs_dim, batch_size)
        self.alpha = alpha
        self.eps = eps
        self.tree = _SumTree(capacity)
        self.max_priority = 1.0

    def push(self, t: Transition) -> None:
        idx = self.position
        super().push(t)
        # New transitions enter at max priority so they're sampled at least once.
        self.tree.set(idx, self.max_priority**self.alpha)

    def sample(self, beta: float = 0.4) -> dict | None:
        if self.size < self.batch_size:
            return None
        total = self.tree.total()
        # Stratified sampling: one draw per equal-width segment.
        segment = total / self.batch_size
        offsets = (np.arange(self.batch_size) + np.random.random(self.batch_size)) * segment
        idx = self.tree.find_batch(offsets)
        idx = np.clip(idx, 0, self.size - 1)
        priorities = self.tree.tree[idx + self.tree.capacity]

        probs = priorities / total
        weights = (self.size * probs) ** (-beta)
        weights /= weights.max()

        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
            "discounts": self.discounts[idx],
            "weights": weights.astype(np.float32),
            "indices": idx,
        }

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        prios = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, float(prios.max()))
        self.tree.set_batch(np.asarray(indices, dtype=np.int64), prios**self.alpha)
