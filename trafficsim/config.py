from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    grid_size: int = 1
    lane_length: int = 12
    queue_window: int = 8
    spawn_rate: float = 0.35
    max_cars: int = 512
    yellow_steps: int = 2
    min_green_steps: int = 4
    max_green_steps: int = 30
    straight_bias: float = 0.55
    # Penalize each phase change so the agent doesn't flicker the light;
    # yellow clearance already costs throughput, this discourages needless toggles.
    switch_penalty: float = 1.0


@dataclass(frozen=True)
class TrainConfig:
    episodes: int = 600
    max_steps: int = 1000
    batch_size: int = 128
    buffer_size: int = 131_072  # 2^17; power of two for the PER sum-tree
    gamma: float = 0.99
    lr: float = 5e-4
    target_sync: int = 500
    tau: float = 0.01
    grad_clip: float = 10.0
    learning_starts: int = 2000
    reward_scale: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    hidden_dims: tuple[int, ...] = (128, 128)
    double_dqn: bool = True
    dueling: bool = True
    # Multi-step returns: propagate delayed congestion reward faster.
    n_step: int = 2
    # Prioritized experience replay.
    prioritized: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    # Robust checkpoint selection: average greedy eval over several seeds.
    eval_seeds: int = 8
    updates_per_step: int = 1
    checkpoint_dir: str = "checkpoints"
    log_every: int = 10


@dataclass(frozen=True)
class RenderConfig:
    cell_size: int = 28
    fps: int = 30
    hud_width: int = 220
    max_grid_px: int = 960
