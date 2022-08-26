from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from ._validation import Validation

T = TypeVar("T", bound=npt.NBitBase)


class Source(Protocol):
    """A radiation source for probing samples."""


@dataclass(frozen=True)
class EmitterResponse(Validation, Generic[T]):
    """Models the response of single fluorescent emitters to a radiation source."""

    x: np.floating[T]
    y: np.floating[T]
    z: np.floating[T]
    photons: int
    wavelength: float

    def validate_photons(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("photons cannot be less than zero")
        return value

    def validate_wavelength(self, value: float, **_) -> float:
        if value <= 0:
            raise ValueError("wavelength must be greater than or equal to zero")
        return value


SampleResponse = List[EmitterResponse]


class Sample(Protocol):
    def response(self, source: Source, dt: float) -> Optional[SampleResponse]:
        pass


class PSF(Protocol, Generic[T]):
    """The point spread function is a response of an optical system to a single point source."""

    def bin(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: np.floating[T], y0: np.floating[T]
    ) -> np.ndarray:
        """Returns the proportion of a normalized PSF centered at (x0, y0) that intersects a square pixel at (x, y).

        Units are in pixels. The origin of the coordinate system lies at the center of a pixel.

        """

    def sample(
        self, x: npt.ArrayLike, y: npt.ArrayLike, x0: np.floating[T], y0: np.floating[T]
    ) -> np.ndarray:
        """Returns samples of the normalized PSF centered at (x0, y0) from the points in x and y."""


OpticsResponse = np.ndarray


class Optics(Protocol):
    """Models an optical system such as a microscope."""

    psf: PSF

    def response(
        self,
        x_lim: Tuple[int, int],
        y_lim: Tuple[int, int],
        sample_response: SampleResponse,
    ) -> OpticsResponse:
        """Computes the response of the optical system to a collection of emitters."""


DetectorResponse = np.ndarray


class Detector(Protocol):
    """Models a detector, such as a camera."""

    def response(self, photons: OpticsResponse, **kwargs) -> DetectorResponse:
        """Computes the response of the detector to an optical system."""
