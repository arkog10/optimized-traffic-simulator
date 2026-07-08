from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pygame

from trafficsim.config import RenderConfig, SimConfig
from trafficsim.grid import Direction, LaneKind, Phase
from trafficsim.sim import TrafficSim


@dataclass(frozen=True)
class Palette:
    bg: tuple[int, int, int] = (17, 20, 26)
    road: tuple[int, int, int] = (45, 50, 58)
    road_edge: tuple[int, int, int] = (32, 36, 42)
    dash: tuple[int, int, int] = (90, 96, 108)
    intersection: tuple[int, int, int] = (58, 64, 74)
    hud_bg: tuple[int, int, int] = (24, 28, 36)
    hud_text: tuple[int, int, int] = (220, 226, 235)
    hud_muted: tuple[int, int, int] = (140, 148, 162)
    light_red: tuple[int, int, int] = (230, 70, 70)
    light_yellow: tuple[int, int, int] = (240, 190, 60)
    light_green: tuple[int, int, int] = (70, 210, 120)


class TrafficRenderer:
    def __init__(
        self,
        sim_config: SimConfig,
        render_config: RenderConfig | None = None,
    ):
        self.sim_config = sim_config
        self.render_config = render_config or RenderConfig()
        self.palette = Palette()
        self.hud_w = self.render_config.hud_width
        self.block_cells = sim_config.lane_length * 2 + 3
        blocks = sim_config.grid_size
        total_cells = blocks * self.block_cells
        # Shrink cells on multi-intersection grids so the full network fits on screen.
        self.cell = min(
            self.render_config.cell_size,
            max(10, self.render_config.max_grid_px // total_cells),
        )
        self.grid_px = total_cells * self.cell
        self.width = self.grid_px + self.hud_w
        self.height = self.grid_px
        self._screen: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._font_sm: pygame.font.Font | None = None
        self._clock = pygame.time.Clock()

    def ensure_ready(self, mode: str = "human") -> None:
        self._ensure_init(mode)

    def render(
        self,
        sim: TrafficSim,
        mode: str = "human",
        *,
        overlay: str | None = None,
        hud_extra: list[str] | None = None,
    ) -> np.ndarray:
        self._ensure_init(mode)
        assert self._screen is not None

        self._screen.fill(self.palette.bg)
        canvas = self._screen.subsurface(pygame.Rect(0, 0, self.grid_px, self.grid_px))
        canvas.fill(self.palette.bg)

        self._draw_roads(sim, canvas)
        self._draw_cars(sim, canvas)
        self._draw_lights(sim, canvas)
        self._draw_hud(sim, hud_extra=hud_extra)
        if overlay:
            self._draw_overlay(overlay)

        if mode == "human":
            pygame.display.flip()
            self._clock.tick(self.render_config.fps)
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self._screen)), (1, 0, 2)
            )

        return np.transpose(np.array(pygame.surfarray.pixels3d(self._screen)), (1, 0, 2))

    def tick_dt(self) -> float:
        """Seconds since last render frame (call after render)."""
        return self._clock.get_time() / 1000.0

    def close(self) -> None:
        if self._screen is not None:
            pygame.display.quit()
            pygame.quit()
        self._screen = None

    def _ensure_init(self, mode: str) -> None:
        if self._screen is not None:
            return
        pygame.init()
        flags = 0 if mode == "human" else pygame.HIDDEN
        self._screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("RL Traffic Grid")
        self._font = pygame.font.SysFont("Menlo", 18)
        self._font_sm = pygame.font.SysFont("Menlo", 14)

    def _origin(self, row: int, col: int) -> tuple[int, int]:
        base_x = col * self.block_cells * self.cell
        base_y = row * self.block_cells * self.cell
        return base_x + self.cell, base_y + self.cell

    def _draw_roads(self, sim: TrafficSim, canvas: pygame.Surface) -> None:
        for intersection in sim.intersections:
            ox, oy = self._origin(intersection.row, intersection.col)
            center = self.sim_config.lane_length + 1
            ix = ox + center * self.cell
            iy = oy + center * self.cell
            arm = self.sim_config.lane_length * self.cell
            road_w = self.cell * 2

            pygame.draw.rect(
                canvas,
                self.palette.intersection,
                pygame.Rect(ix - road_w // 2, iy - road_w // 2, road_w, road_w),
            )
            pygame.draw.rect(
                canvas,
                self.palette.road,
                pygame.Rect(ix - road_w // 4, oy, road_w // 2, arm + road_w // 2),
            )
            pygame.draw.rect(
                canvas,
                self.palette.road,
                pygame.Rect(ix - road_w // 4, iy, road_w // 2, arm + road_w // 2),
            )
            pygame.draw.rect(
                canvas,
                self.palette.road,
                pygame.Rect(ox, iy - road_w // 4, arm + road_w // 2, road_w // 2),
            )
            pygame.draw.rect(
                canvas,
                self.palette.road,
                pygame.Rect(ix, iy - road_w // 4, arm + road_w // 2, road_w // 2),
            )

            for offset in (self.cell // 2,):
                self._dashed_line(
                    canvas,
                    (ix, oy + offset),
                    (ix, oy + arm),
                    self.palette.dash,
                )
                self._dashed_line(
                    canvas,
                    (ix, iy + road_w // 2),
                    (ix, iy + arm),
                    self.palette.dash,
                )
                self._dashed_line(
                    canvas,
                    (ox + offset, iy),
                    (ox + arm, iy),
                    self.palette.dash,
                )
                self._dashed_line(
                    canvas,
                    (ix + road_w // 2, iy),
                    (ix + arm, iy),
                    self.palette.dash,
                )

    def _draw_cars(self, sim: TrafficSim, canvas: pygame.Surface) -> None:
        for lane in sim.lanes:
            ox, oy = self._origin(
                sim.intersections[lane.intersection_id].row,
                sim.intersections[lane.intersection_id].col,
            )
            center = self.sim_config.lane_length + 1
            for idx, car_id in enumerate(lane.cells):
                if car_id < 0:
                    continue
                wait = int(sim.cars.wait_steps[car_id])
                color = self._wait_color(wait)
                rect = self._car_rect(lane.direction, lane.kind, ox, oy, center, idx)
                pygame.draw.rect(canvas, color, rect, border_radius=4)

    def _draw_lights(self, sim: TrafficSim, canvas: pygame.Surface) -> None:
        for intersection in sim.intersections:
            ox, oy = self._origin(intersection.row, intersection.col)
            center = self.sim_config.lane_length + 1
            ix = ox + center * self.cell
            iy = oy + center * self.cell
            radius = max(4, self.cell // 5)

            for direction in Direction:
                color = self._light_color(intersection.phase, direction)
                lx, ly = self._light_pos(direction, ix, iy, radius)
                pygame.draw.circle(canvas, color, (lx, ly), radius)
                glow = pygame.Surface((radius * 4, radius * 4), pygame.SRCALPHA)
                pygame.draw.circle(
                    glow,
                    (*color, 70),
                    (radius * 2, radius * 2),
                    radius * 2,
                )
                canvas.blit(glow, (lx - radius * 2, ly - radius * 2))

    def _draw_hud(self, sim: TrafficSim, hud_extra: list[str] | None = None) -> None:
        assert self._screen is not None and self._font is not None
        assert self._font_sm is not None

        panel = pygame.Surface((self.hud_w, self.height), pygame.SRCALPHA)
        panel.fill((*self.palette.hud_bg, 220))
        self._screen.blit(panel, (self.grid_px, 0))

        metrics = sim.metrics
        avg_wait = metrics.total_wait / max(metrics.step, 1)
        lines = [
            ("RL Traffic Grid", self.palette.hud_text, self._font),
            ("", self.palette.hud_muted, self._font_sm),
            (f"Step      {metrics.step}", self.palette.hud_muted, self._font_sm),
            (f"Active    {metrics.active_cars}", self.palette.hud_muted, self._font_sm),
            (f"Stopped   {metrics.stopped_cars}", self.palette.hud_muted, self._font_sm),
            (f"Avg wait  {avg_wait:.2f}", self.palette.hud_muted, self._font_sm),
            (f"Completed {metrics.completed}", self.palette.hud_muted, self._font_sm),
        ]
        if hud_extra:
            lines.append(("", self.palette.hud_muted, self._font_sm))
            for row in hud_extra:
                lines.append((row, self.palette.hud_muted, self._font_sm))

        y = 20
        for text, color, font in lines:
            if not text:
                y += 8
                continue
            surf = font.render(text, True, color)
            self._screen.blit(surf, (self.grid_px + 16, y))
            y += surf.get_height() + 8

    def _draw_overlay(self, message: str) -> None:
        assert self._screen is not None and self._font is not None
        lines = message.split("\n")
        line_height = self._font.get_height() + 6
        box_h = 24 + line_height * len(lines)
        box_w = min(self.width - 40, 420)
        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((16, 20, 28, 210))
        pygame.draw.rect(box, self.palette.hud_muted, box.get_rect(), width=1, border_radius=8)

        y = 12
        for line in lines:
            surf = self._font.render(line, True, self.palette.hud_text)
            box.blit(surf, (16, y))
            y += line_height

        x = (self.width - box_w) // 2
        y = (self.height - box_h) // 2
        self._screen.blit(box, (x, y))

    def _car_rect(
        self,
        direction: Direction,
        kind: LaneKind,
        ox: int,
        oy: int,
        center: int,
        idx: int,
    ) -> pygame.Rect:
        lane_w = self.cell
        car_len = max(8, self.cell - 6)
        car_w = max(8, lane_w - 10)
        margin = 3

        if kind == LaneKind.INBOUND:
            if direction == Direction.NORTH:
                x = ox + center * self.cell + margin
                y = oy + (self.sim_config.lane_length - 1 - idx) * self.cell + margin
                return pygame.Rect(x, y, car_w, car_len)
            if direction == Direction.SOUTH:
                x = ox + center * self.cell + margin
                y = oy + (center + 1 + idx) * self.cell + margin
                return pygame.Rect(x, y, car_w, car_len)
            if direction == Direction.WEST:
                x = ox + (self.sim_config.lane_length - 1 - idx) * self.cell + margin
                y = oy + center * self.cell + margin
                return pygame.Rect(x, y, car_len, car_w)
            x = ox + (center + 1 + idx) * self.cell + margin
            y = oy + center * self.cell + margin
            return pygame.Rect(x, y, car_len, car_w)

        if direction == Direction.NORTH:
            x = ox + center * self.cell + margin
            y = oy + (center - 1 - idx) * self.cell + margin
            return pygame.Rect(x, y, car_w, car_len)
        if direction == Direction.SOUTH:
            x = ox + center * self.cell + margin
            y = oy + (center + 1 + idx) * self.cell + margin
            return pygame.Rect(x, y, car_w, car_len)
        if direction == Direction.WEST:
            x = ox + (center - 1 - idx) * self.cell + margin
            y = oy + center * self.cell + margin
            return pygame.Rect(x, y, car_len, car_w)
        x = ox + (center + 1 + idx) * self.cell + margin
        y = oy + center * self.cell + margin
        return pygame.Rect(x, y, car_len, car_w)

    def _wait_color(self, wait: int) -> tuple[int, int, int]:
        t = min(wait / 20.0, 1.0)
        low = np.array([70, 210, 120])
        mid = np.array([240, 190, 60])
        high = np.array([230, 90, 70])
        if t < 0.5:
            rgb = low * (1 - t * 2) + mid * (t * 2)
        else:
            rgb = mid * (2 - t * 2) + high * (t * 2 - 1)
        return tuple(int(v) for v in rgb)

    def _light_color(self, phase: Phase, direction: Direction) -> tuple[int, int, int]:
        if phase == Phase.ALL_RED:
            return self.palette.light_red
        if phase == Phase.NS_GREEN and direction in (Direction.NORTH, Direction.SOUTH):
            return self.palette.light_green
        if phase == Phase.EW_GREEN and direction in (Direction.EAST, Direction.WEST):
            return self.palette.light_green
        return self.palette.light_red

    def _light_pos(
        self, direction: Direction, ix: int, iy: int, radius: int
    ) -> tuple[int, int]:
        offset = self.cell
        if direction == Direction.NORTH:
            return ix + self.cell // 2, iy + offset
        if direction == Direction.SOUTH:
            return ix + self.cell // 2, iy + offset * 2
        if direction == Direction.WEST:
            return ix + offset, iy + self.cell // 2
        return ix + offset * 2, iy + self.cell // 2

    def _dashed_line(
        self,
        surface: pygame.Surface,
        start: tuple[int, int],
        end: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        x1, y1 = start
        x2, y2 = end
        dash = self.cell // 2
        gap = self.cell // 3
        if x1 == x2:
            y = min(y1, y2)
            stop = max(y1, y2)
            while y < stop:
                pygame.draw.line(surface, color, (x1, y), (x2, min(y + dash, stop)), 1)
                y += dash + gap
        else:
            x = min(x1, x2)
            stop = max(x1, x2)
            while x < stop:
                pygame.draw.line(surface, color, (x, y1), (min(x + dash, stop), y2), 1)
                x += dash + gap
