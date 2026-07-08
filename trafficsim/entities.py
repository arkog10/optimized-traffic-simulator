from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trafficsim.config import SimConfig
from trafficsim.grid import Direction, Phase


@dataclass
class CarPool:
    lane_id: np.ndarray
    cell_pos: np.ndarray
    wait_steps: np.ndarray
    turn: np.ndarray
    active: np.ndarray
    next_free: int = 1

    @classmethod
    def create(cls, max_cars: int) -> CarPool:
        return cls(
            lane_id=np.full(max_cars, -1, dtype=np.int32),
            cell_pos=np.full(max_cars, -1, dtype=np.int32),
            wait_steps=np.zeros(max_cars, dtype=np.int32),
            turn=np.zeros(max_cars, dtype=np.int32),
            active=np.zeros(max_cars, dtype=bool),
        )

    def spawn(
        self,
        lane_id: int,
        cell_pos: int,
        turn: int,
        rng: np.random.Generator,
    ) -> int | None:
        if self.next_free >= len(self.active):
            return None
        car_id = self.next_free
        self.next_free += 1
        self.lane_id[car_id] = lane_id
        self.cell_pos[car_id] = cell_pos
        self.wait_steps[car_id] = 0
        self.turn[car_id] = turn
        self.active[car_id] = True
        return car_id

    def deactivate(self, car_id: int) -> None:
        self.active[car_id] = False
        self.lane_id[car_id] = -1
        self.cell_pos[car_id] = -1
        self.wait_steps[car_id] = 0


@dataclass
class SimMetrics:
    step: int = 0
    stopped_cars: int = 0
    total_wait: int = 0
    completed: int = 0
    active_cars: int = 0
    per_intersection_stopped: dict[int, int] = field(default_factory=dict)


def phase_allows(direction: Direction, phase: Phase) -> bool:
    if phase == Phase.ALL_RED:
        return False
    if phase == Phase.NS_GREEN:
        return direction in (Direction.NORTH, Direction.SOUTH)
    return direction in (Direction.EAST, Direction.WEST)


def sample_turn(straight_bias: float, rng: np.random.Generator) -> int:
    roll = rng.random()
    if roll < straight_bias:
        return 0
    if roll < straight_bias + (1 - straight_bias) / 2:
        return 1
    return 2
