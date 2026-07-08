from __future__ import annotations

SPEED_PRESETS = (1, 2, 4, 6, 8, 12)


class SpeedController:
    """Sim steps per second — decoupled from render FPS."""

    def __init__(self, preset_index: int = 1):
        self.preset_index = preset_index
        self._accumulator = 0.0

    @property
    def steps_per_second(self) -> float:
        return float(SPEED_PRESETS[self.preset_index])

    @property
    def label(self) -> str:
        sps = self.steps_per_second
        if sps < 1:
            return f"{sps:.1f} steps/s"
        return f"{int(sps)} steps/s"

    def faster(self) -> None:
        self.preset_index = min(self.preset_index + 1, len(SPEED_PRESETS) - 1)

    def slower(self) -> None:
        self.preset_index = max(self.preset_index - 1, 0)

    def tick(self, dt_seconds: float) -> int:
        self._accumulator += self.steps_per_second * dt_seconds
        steps = int(self._accumulator)
        self._accumulator -= steps
        return steps

    def reset(self) -> None:
        self._accumulator = 0.0
