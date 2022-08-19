from typing import Protocol

import numpy as np
import numpy.typing as npt
from scipy import special  # type: ignore

from leb.imajin import DEFAULT_FLOAT_TYPE


class PSF(Protocol):
    def bin(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: float = 0.0, y0: float = 0.0
    ) -> np.ndarray:
        """Returns the proportion of a normalized PSF centered at (x0, y0) that intersects a square pixel at (x, y).

        Units are in pixels. The origin of the coordinate system lies at the center of a pixel.

        """

    def sample(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: float = 0, y0: float = 0.0
    ) -> np.ndarray:
        """Returns samples of the normalized PSF centered at (x0, y0) from the points in x and y."""


class Gaussian2D(PSF):
    def __init__(self, fwhm: float = 1):
        self.fwhm = fwhm

    def __repr__(self) -> str:
        return f"Gaussian2D(fwhm={self.fwhm})"

    @property
    def fwhm(self) -> float:
        return self._fwhm

    @fwhm.setter
    def fwhm(self, value: float) -> None:
        if value <= 0:
            raise ValueError("fwhm must be greater than 0")

        self._fwhm = value

    def bin(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: float = 0.0, y0: float = 0.0
    ) -> np.ndarray:
        x0_, y0_ = DEFAULT_FLOAT_TYPE(x0), DEFAULT_FLOAT_TYPE(y0)
        x, y = np.asanyarray(x), np.asanyarray(y)
        scale = np.sqrt(2) * self.fwhm / 2.3548
        binned_values: np.ndarray = (
            0.25
            * (
                special.erf((x - x0_ + 0.5) / scale)
                - special.erf((x + x0_ - 0.5) / scale)
            )
            * (
                special.erf((y - y0_ + 0.5) / scale)
                - special.erf((y + y0_ - 0.5) / scale)
            )
        )
        return binned_values

    def sample(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: float = 0, y0: float = 0.0
    ) -> np.ndarray:
        x0_, y0_ = DEFAULT_FLOAT_TYPE(x0), DEFAULT_FLOAT_TYPE(y0)
        x, y = np.asanyarray(x), np.asanyarray(y)
        sigma = self.fwhm / 2.3548
        samples: np.ndarray = (
            1.0
            / (2.0 * np.pi * sigma * sigma)
            * np.exp(
                -(
                    (x - x0_) ** 2.0 / (2.0 * sigma**2.0)
                    + (y - y0_) ** 2.0 / (2.0 * sigma**2.0)
                )
            )
        )
        return samples
