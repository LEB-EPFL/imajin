from typing import Protocol, Tuple

import numpy as np
import numpy.typing as npt

from leb.imajin.samples import SampleResponse


class PSF(Protocol):
    """The point spread function is a response of an optical system to a single point source."""

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


class OpticalSystem(Protocol):
    """Models an optical system such as a microscope."""

    psf: PSF

    def response(
        self,
        x_lim: Tuple[int, int],
        y_lim: Tuple[int, int],
        sample_response: SampleResponse,
    ) -> np.ndarray:
        """Computes the response of the optical system to a collection of emitters."""
