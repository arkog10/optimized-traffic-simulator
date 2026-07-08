from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trafficsim.config import SimConfig
from trafficsim.entities import CarPool, SimMetrics, phase_allows, sample_turn
from trafficsim.grid import (
    Direction,
    Intersection,
    Lane,
    LaneKind,
    Phase,
    build_grid,
    inbound_lane_for,
    neighbor_intersection,
    outbound_lane_for,
    turn_direction,
)


@dataclass
class StepResult:
    observations: np.ndarray
    reward: float
    stopped_cars: int
    per_intersection_stopped: dict[int, int]
    phase_switches: int


class TrafficSim:
    def __init__(self, config: SimConfig, seed: int | None = None):
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.lanes, self.intersections, self.lane_lookup = build_grid(config)
        self.cars = CarPool.create(config.max_cars)
        self.metrics = SimMetrics()
        self._lane_by_id = {lane.lane_id: lane for lane in self.lanes}
        self._intersection_lanes = self._group_lanes_by_intersection()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        for lane in self.lanes:
            lane.cells.fill(-1)
        for intersection in self.intersections:
            intersection.phase = Phase.NS_GREEN
            intersection.time_in_phase = 0
            intersection.yellow_timer = 0
            intersection.pending_phase = None
        self.cars = CarPool.create(self.config.max_cars)
        self.metrics = SimMetrics()
        return self.get_observations()

    def step(self, actions: np.ndarray | int | None = None) -> StepResult:
        if actions is None:
            actions = np.zeros(len(self.intersections), dtype=np.int32)
        elif isinstance(actions, (int, np.integer)):
            actions = np.array([int(actions)], dtype=np.int32)
        else:
            actions = np.asarray(actions, dtype=np.int32)

        phase_switches = self._apply_actions(actions)
        self._update_lights()
        stopped = self._move_all_lanes()
        self._spawn_cars()
        self._update_metrics(stopped)

        reward = -float(stopped)
        if self.config.switch_penalty > 0 and phase_switches > 0:
            reward -= self.config.switch_penalty * phase_switches

        per_intersection = {
            i: self.metrics.per_intersection_stopped.get(i, 0)
            for i in range(len(self.intersections))
        }
        return StepResult(
            observations=self.get_observations(),
            reward=reward,
            stopped_cars=stopped,
            per_intersection_stopped=per_intersection,
            phase_switches=phase_switches,
        )

    def get_observations(self) -> np.ndarray:
        obs_list = []
        for intersection in self.intersections:
            obs_list.append(self._intersection_observation(intersection))
        return np.array(obs_list, dtype=np.float32)

    def snapshot(self) -> dict:
        return {
            "lanes": [
                {
                    "lane_id": lane.lane_id,
                    "direction": int(lane.direction),
                    "kind": int(lane.kind),
                    "cells": lane.cells.copy(),
                    "intersection_id": lane.intersection_id,
                    "is_source": lane.is_source,
                    "is_sink": lane.is_sink,
                }
                for lane in self.lanes
            ],
            "intersections": [
                {
                    "intersection_id": ix.intersection_id,
                    "row": ix.row,
                    "col": ix.col,
                    "phase": int(ix.phase),
                    "time_in_phase": ix.time_in_phase,
                }
                for ix in self.intersections
            ],
            "metrics": {
                "step": self.metrics.step,
                "stopped_cars": self.metrics.stopped_cars,
                "total_wait": self.metrics.total_wait,
                "completed": self.metrics.completed,
                "active_cars": self.metrics.active_cars,
            },
            "cars": {
                "wait_steps": self.cars.wait_steps.copy(),
                "active": self.cars.active.copy(),
                "lane_id": self.cars.lane_id.copy(),
                "cell_pos": self.cars.cell_pos.copy(),
            },
            "grid_size": self.config.grid_size,
            "lane_length": self.config.lane_length,
        }

    def _group_lanes_by_intersection(self) -> dict[int, list[Lane]]:
        grouped: dict[int, list[Lane]] = {}
        for lane in self.lanes:
            grouped.setdefault(lane.intersection_id, []).append(lane)
        return grouped

    def _apply_actions(self, actions: np.ndarray) -> int:
        switches = 0
        for intersection, action in zip(self.intersections, actions, strict=True):
            target = Phase.NS_GREEN if int(action) == 0 else Phase.EW_GREEN
            if intersection.phase == Phase.ALL_RED:
                continue
            current = (
                Phase.NS_GREEN
                if intersection.phase == Phase.NS_GREEN
                else Phase.EW_GREEN
            )
            if target == current:
                continue
            if intersection.time_in_phase < self.config.min_green_steps:
                continue
            if intersection.pending_phase != target:
                intersection.pending_phase = target
                switches += 1
        return switches

    def _update_lights(self) -> None:
        for intersection in self.intersections:
            if intersection.phase == Phase.ALL_RED:
                intersection.yellow_timer -= 1
                if intersection.yellow_timer <= 0:
                    intersection.phase = intersection.pending_phase or Phase.NS_GREEN
                    intersection.pending_phase = None
                    intersection.time_in_phase = 0
                continue

            if intersection.pending_phase is not None:
                intersection.phase = Phase.ALL_RED
                intersection.yellow_timer = self.config.yellow_steps
                continue

            if (
                self.config.max_green_steps > 0
                and intersection.time_in_phase >= self.config.max_green_steps
            ):
                other = (
                    Phase.EW_GREEN
                    if intersection.phase == Phase.NS_GREEN
                    else Phase.NS_GREEN
                )
                intersection.pending_phase = other
                continue

            intersection.time_in_phase += 1

    def _move_all_lanes(self) -> int:
        stopped = 0
        for lane in self.lanes:
            if lane.kind == LaneKind.INBOUND:
                stopped += self._move_inbound_lane(lane)
            else:
                self._move_outbound_lane(lane)
        return stopped

    def _move_inbound_lane(self, lane: Lane) -> int:
        stopped = 0
        intersection = self.intersections[lane.intersection_id]
        cells = lane.cells

        for idx in range(lane.stop_line, -1, -1):
            car_id = int(cells[idx])
            if car_id < 0:
                continue

            if idx == lane.stop_line:
                if not phase_allows(lane.direction, intersection.phase):
                    self.cars.wait_steps[car_id] += 1
                    stopped += 1
                    continue
                if not self._cross_intersection(car_id, lane):
                    self.cars.wait_steps[car_id] += 1
                    stopped += 1
                continue

            next_idx = idx + 1
            if cells[next_idx] >= 0:
                self.cars.wait_steps[car_id] += 1
                stopped += 1
                continue

            cells[next_idx] = car_id
            cells[idx] = -1
            self.cars.cell_pos[car_id] = next_idx
            self.cars.lane_id[car_id] = lane.lane_id

        return stopped

    def _move_outbound_lane(self, lane: Lane) -> None:
        cells = lane.cells
        length = len(cells)

        for idx in range(length - 2, -1, -1):
            car_id = int(cells[idx])
            if car_id < 0:
                continue

            next_idx = idx + 1
            if cells[next_idx] >= 0:
                self.cars.wait_steps[car_id] += 1
                continue

            if next_idx == length - 1 and lane.is_sink:
                cells[idx] = -1
                self.cars.deactivate(car_id)
                self.metrics.completed += 1
                continue

            cells[next_idx] = car_id
            cells[idx] = -1
            self.cars.cell_pos[car_id] = next_idx
            self.cars.lane_id[car_id] = lane.lane_id

    def _cross_intersection(self, car_id: int, inbound_lane: Lane) -> bool:
        intersection = self.intersections[inbound_lane.intersection_id]
        turn = int(self.cars.turn[car_id])
        exit_direction = turn_direction(inbound_lane.direction, turn)

        neighbor = neighbor_intersection(
            intersection.row,
            intersection.col,
            exit_direction,
            self.config.grid_size,
        )

        if neighbor is None:
            outbound_id = outbound_lane_for(
                self.lane_lookup,
                intersection.row,
                intersection.col,
                exit_direction,
            )
            outbound = self._lane_by_id[outbound_id]
        else:
            nrow, ncol = neighbor
            approach_arm = _opposite(exit_direction)
            outbound_id = inbound_lane_for(
                self.lane_lookup, nrow, ncol, approach_arm
            )
            outbound = self._lane_by_id[outbound_id]
        if outbound.cells[0] >= 0:
            return False

        inbound_lane.cells[inbound_lane.stop_line] = -1
        outbound.cells[0] = car_id
        self.cars.lane_id[car_id] = outbound.lane_id
        self.cars.cell_pos[car_id] = 0
        return True

    def _spawn_cars(self) -> None:
        for lane in self.lanes:
            if lane.kind != LaneKind.INBOUND or not lane.is_source:
                continue
            if lane.cells[0] >= 0:
                continue
            if self.rng.random() > self.config.spawn_rate:
                continue
            turn = sample_turn(self.config.straight_bias, self.rng)
            car_id = self.cars.spawn(lane.lane_id, 0, turn, self.rng)
            if car_id is None:
                continue
            lane.cells[0] = car_id

    def _intersection_observation(self, intersection: Intersection) -> np.ndarray:
        queues = []
        for direction in Direction:
            lane_id = inbound_lane_for(
                self.lane_lookup, intersection.row, intersection.col, direction
            )
            lane = self._lane_by_id[lane_id]
            count = 0
            start = max(0, lane.stop_line - self.config.queue_window + 1)
            for idx in range(start, lane.stop_line + 1):
                if lane.cells[idx] >= 0:
                    count += 1
            queues.append(count)

        if intersection.phase == Phase.NS_GREEN:
            phase_vec = [1.0, 0.0, 0.0]
        elif intersection.phase == Phase.EW_GREEN:
            phase_vec = [0.0, 1.0, 0.0]
        else:
            phase_vec = [0.0, 0.0, 1.0]

        norm_time = min(
            intersection.time_in_phase / max(self.config.min_green_steps * 4, 1),
            1.0,
        )
        return np.array(
            queues + phase_vec + [norm_time],
            dtype=np.float32,
        )

    def _update_metrics(self, stopped: int) -> None:
        self.metrics.step += 1
        self.metrics.stopped_cars = stopped
        self.metrics.total_wait += stopped
        self.metrics.active_cars = int(self.cars.active.sum())
        self.metrics.per_intersection_stopped = self._stopped_by_intersection()

    def _stopped_by_intersection(self) -> dict[int, int]:
        counts = {ix.intersection_id: 0 for ix in self.intersections}
        for lane in self.lanes:
            if lane.kind != LaneKind.INBOUND:
                continue
            intersection = self.intersections[lane.intersection_id]
            for idx in range(lane.stop_line + 1):
                car_id = int(lane.cells[idx])
                if car_id < 0:
                    continue
                if idx == lane.stop_line and not phase_allows(
                    lane.direction, intersection.phase
                ):
                    counts[intersection.intersection_id] += 1
                elif idx < lane.stop_line and lane.cells[idx + 1] >= 0:
                    counts[intersection.intersection_id] += 1
        return counts


def _opposite(direction: Direction) -> Direction:
    return Direction((direction + 2) % 4)
