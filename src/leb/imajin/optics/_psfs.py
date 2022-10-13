from typing import Generic, TypeVar

import numpy as np
import numpy.typing as npt
from scipy import special  # type: ignore

from leb.imajin import PSF

T = TypeVar("T", bound=npt.NBitBase)


class Gaussian2D(PSF, Generic[T]):
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
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        x0: np.floating[T],
        y0: np.floating[T],
        dx: float = 1,
        dy: float = 1,
    ) -> np.ndarray:
        """Compute the integrated number of photons across a pixel from an emitter at (x0, y0).

        Parameters
        ----------
        x: numpy.typing.ArrayLike
        y: numpy.typing.ArrayLike
        x0: numpy.floating
        y0: numpy.floating
        dx: float
            The size of a pixel in the x-dirction. (Default: 1.0)
        dy: float
            The size of a pixel in the y-direction. (Default: 1.0)
        """
        x, y = np.asanyarray(x), np.asanyarray(y)
        scale = np.sqrt(2) * self.fwhm / 2.3548
        binned_values: np.ndarray = (
            0.25
            * (special.erf((x - x0 + dx) / scale) - special.erf((x - x0) / scale))
            * (special.erf((y - y0 + dy) / scale) - special.erf((y - y0) / scale))
        )
        return binned_values

    def sample(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: np.floating[T], y0: np.floating[T]
    ) -> np.ndarray:
        x, y = np.asanyarray(x), np.asanyarray(y)
        sigma = self.fwhm / 2.3548
        samples: np.ndarray = (
            1.0
            / (2.0 * np.pi * sigma * sigma)
            * np.exp(
                -((x - x0) ** 2.0 / (2.0 * sigma**2.0) + (y - y0) ** 2.0 / (2.0 * sigma**2.0))
            )
        )
        return samples
