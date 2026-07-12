from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class ControlTick:
    """One controller interval expressed in exact MuJoCo physics steps."""

    index: int
    substeps: int
    duration_s: float
    scheduled_end_s: float


class PhaseAccumulatorClock:
    """Dither integer physics steps to preserve a requested control rate.

    MuJoCo advances by a fixed timestep, while a controller period frequently
    is not an integer multiple of it (120 Hz with a 2 ms model timestep, for
    example).  Rounding once produces 125 Hz.  Rounding cumulative phase gives
    a 4/4/5-step pattern whose long-run rate converges to exactly 120 Hz.
    """

    def __init__(self, physics_dt_s: float, control_hz: float) -> None:
        self.physics_dt_s = float(physics_dt_s)
        self.control_hz = float(control_hz)
        if not math.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0.0:
            raise ValueError("physics timestep must be positive and finite")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive and finite")
        physics_hz = 1.0 / self.physics_dt_s
        if self.control_hz > physics_hz * (1.0 + 1e-12):
            raise ValueError(
                f"control_hz ({self.control_hz:g}) cannot exceed MuJoCo physics rate ({physics_hz:g} Hz)"
            )
        self._ticks = 0
        self._physics_steps = 0

    @property
    def ticks(self) -> int:
        return self._ticks

    @property
    def physics_steps(self) -> int:
        return self._physics_steps

    def reset(self) -> None:
        self._ticks = 0
        self._physics_steps = 0

    def next_tick(self) -> ControlTick:
        next_ticks = self._ticks + 1
        exact_cumulative_steps = next_ticks / (self.control_hz * self.physics_dt_s)
        # Avoid Python's ties-to-even round: nearest-step phase is easier to
        # reason about and remains deterministic across platforms.
        cumulative_steps = int(math.floor(exact_cumulative_steps + 0.5))
        substeps = cumulative_steps - self._physics_steps
        if substeps < 1:
            # The constructor prevents a genuinely faster-than-physics clock;
            # this guard only protects against floating-point edge cases.
            substeps = 1
            cumulative_steps = self._physics_steps + 1
        self._ticks = next_ticks
        self._physics_steps = cumulative_steps
        return ControlTick(
            index=self._ticks - 1,
            substeps=substeps,
            duration_s=substeps * self.physics_dt_s,
            scheduled_end_s=self._ticks / self.control_hz,
        )

    def effective_hz(self) -> float:
        duration = self._physics_steps * self.physics_dt_s
        return 0.0 if duration <= 0.0 else self._ticks / duration
