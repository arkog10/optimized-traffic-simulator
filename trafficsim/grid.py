from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from trafficsim.config import SimConfig


class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3


class Phase(IntEnum):
    NS_GREEN = 0
    EW_GREEN = 1
    ALL_RED = 2


class LaneKind(IntEnum):
    INBOUND = 0
    OUTBOUND = 1


@dataclass
class Lane:
    lane_id: int
    cells: np.ndarray
    direction: Direction
    kind: LaneKind
    stop_line: int
    intersection_id: int
    is_source: bool = False
    is_sink: bool = False


@dataclass
class Intersection:
    intersection_id: int
    row: int
    col: int
    phase: Phase = Phase.NS_GREEN
    time_in_phase: int = 0
    yellow_timer: int = 0
    pending_phase: Phase | None = None


def _opposite(direction: Direction) -> Direction:
    return Direction((direction + 2) % 4)


def _left(direction: Direction) -> Direction:
    return Direction((direction - 1) % 4)


def _right(direction: Direction) -> Direction:
    return Direction((direction + 1) % 4)


def build_grid(config: SimConfig) -> tuple[list[Lane], list[Intersection], np.ndarray]:
    lanes: list[Lane] = []
    intersections: list[Intersection] = []
    lane_id = 0

    for row in range(config.grid_size):
        for col in range(config.grid_size):
            intersections.append(
                Intersection(intersection_id=len(intersections), row=row, col=col)
            )

    for row in range(config.grid_size):
        for col in range(config.grid_size):
            intersection_id = row * config.grid_size + col
            for direction in Direction:
                inbound = Lane(
                    lane_id=lane_id,
                    cells=np.full(config.lane_length, -1, dtype=np.int32),
                    direction=direction,
                    kind=LaneKind.INBOUND,
                    stop_line=config.lane_length - 2,
                    intersection_id=intersection_id,
                    is_source=_is_edge_source(row, col, direction, config.grid_size),
                )
                lane_id += 1
                lanes.append(inbound)

                outbound = Lane(
                    lane_id=lane_id,
                    cells=np.full(config.lane_length, -1, dtype=np.int32),
                    direction=direction,
                    kind=LaneKind.OUTBOUND,
                    stop_line=-1,
                    intersection_id=intersection_id,
                    is_sink=_is_edge_sink(row, col, direction, config.grid_size),
                )
                lane_id += 1
                lanes.append(outbound)

    lane_lookup = np.zeros((config.grid_size, config.grid_size, 4, 2), dtype=np.int32)
    for lane in lanes:
        intersection = intersections[lane.intersection_id]
        lane_lookup[intersection.row, intersection.col, lane.direction, lane.kind] = (
            lane.lane_id
        )

    return lanes, intersections, lane_lookup


def _is_edge_source(row: int, col: int, direction: Direction, grid_size: int) -> bool:
    if direction == Direction.NORTH and row == 0:
        return True
    if direction == Direction.SOUTH and row == grid_size - 1:
        return True
    if direction == Direction.WEST and col == 0:
        return True
    if direction == Direction.EAST and col == grid_size - 1:
        return True
    return False


def _is_edge_sink(row: int, col: int, direction: Direction, grid_size: int) -> bool:
    if direction == Direction.NORTH and row == 0:
        return True
    if direction == Direction.SOUTH and row == grid_size - 1:
        return True
    if direction == Direction.WEST and col == 0:
        return True
    if direction == Direction.EAST and col == grid_size - 1:
        return True
    return False


def inbound_lane_for(lane_lookup: np.ndarray, row: int, col: int, direction: Direction) -> int:
    return int(lane_lookup[row, col, direction, LaneKind.INBOUND])


def outbound_lane_for(lane_lookup: np.ndarray, row: int, col: int, direction: Direction) -> int:
    return int(lane_lookup[row, col, direction, LaneKind.OUTBOUND])


def movement_direction(inbound_arm: Direction) -> Direction:
    return _opposite(inbound_arm)


def exit_arm(inbound_arm: Direction, turn: int) -> Direction:
    facing = movement_direction(inbound_arm)
    if turn == 0:
        return facing
    if turn == 1:
        return _left(facing)
    return _right(facing)


def turn_direction(from_direction: Direction, turn: int) -> Direction:
    return exit_arm(from_direction, turn)


def neighbor_intersection(
    row: int, col: int, direction: Direction, grid_size: int
) -> tuple[int, int] | None:
    if direction == Direction.NORTH and row > 0:
        return row - 1, col
    if direction == Direction.SOUTH and row < grid_size - 1:
        return row + 1, col
    if direction == Direction.WEST and col > 0:
        return row, col - 1
    if direction == Direction.EAST and col < grid_size - 1:
        return row, col + 1
    return None
