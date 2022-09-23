from typing import Generic, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from leb.imajin import Source

T = TypeVar("T", bound=npt.NBitBase)


class UniformMono2D(Source, Generic[T]):
    def __init__(
        self,
        power_max: float,
        power: float,
        x_lim: Tuple[float, float],
        y_lim: Tuple[float, float],
    ):
        self._power_max = power_max
        self.power = power
        self.x_lim = x_lim
        self.y_lim = y_lim

    def __repr__(self) -> str:
        return f"UniformMono2D(power_max={self._power_max}, power={self.power}, x_lim={self.x_lim}, y_lim={self.y_lim})"

    @property
    def POWER_MAX(self) -> float:
        return self._power_max

    @property
    def power(self) -> float:
        return self._power

    @power.setter
    def power(self, value: float) -> None:
        if value < 0:
            raise ValueError("power must be greater than or equal to zero")
        if value > self._power_max:
            raise ValueError("power cannot exceed the maximum power")

        self._power = value

    @property
    def x_lim(self) -> Tuple[float, float]:
        return self._x_lim

    @x_lim.setter
    def x_lim(self, value: Tuple[float, float]) -> None:
        if value[0] >= value[1]:
            raise ValueError(
                "the first value of x_lim must be less than the second value"
            )

        self._x_lim = value

    @property
    def y_lim(self) -> Tuple[float, float]:
        return self._y_lim

    @y_lim.setter
    def y_lim(self, value: Tuple[float, float]) -> None:
        if value[0] >= value[1]:
            raise ValueError(
                "the first value of y_lim must be less than the second value"
            )

        self._y_lim = value

    def e_field(
        self, x: np.floating[T], y: np.floating[T], impedance: float = 1
    ) -> npt.ArrayLike:
        e_field: complex = np.sqrt(
            impedance * self.irradiance(x, y),
            dtype=complex,
        )
        return e_field

    def irradiance(self, x: np.floating[T], y: np.floating[T]) -> float:
        if x < self.x_lim[0] or x > self.x_lim[1]:
            return 0.0
        elif y < self.y_lim[0] or y > self.y_lim[1]:
            return 0.0
        else:
            return (
                self.power
                / (self.x_lim[1] - self.x_lim[0])
                / (self.y_lim[1] - self.y_lim[0])
            )
