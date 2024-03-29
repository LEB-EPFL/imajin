from dataclasses import dataclass
from typing import Generic, List, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from ._validation import Validation

T = TypeVar("T", bound=npt.NBitBase)


class Source(Protocol, Generic[T]):
    """A radiation source for probing samples."""

    def irradiance(self, x: np.floating[T], y: np.floating[T]) -> float:
        """Computes the irradiance of the source at a point in space."""


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
            raise ValueError("photons must be greater than zero")
        return value

    def validate_wavelength(self, value: float, **_) -> float:
        if value <= 0:
            raise ValueError("wavelength must be greater than zero")
        return value


class Emitter(Protocol):
    """A single fluorescence emitter."""

    def response(self, time: float, dt: float, source: Source) -> EmitterResponse:
        pass


SampleResponse = List[EmitterResponse]


class Sample(Protocol):
    def response(self, time: float, dt: float, source: Source) -> SampleResponse:
        raise NotImplementedError


class PSF(Protocol, Generic[T]):
    """The point spread function is a response of an optical system to a single point source."""

    def bin(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        x0: np.floating[T],
        y0: np.floating[T],
        dx: float = 1,
        dy: float = 1,
    ) -> np.ndarray:
        """Returns the proportion of a normalized PSF centered at (x0, y0) that intersects a
        rectangular area at (x, y) with side-lenghts (dx, dy).

        Units are in pixels. The origin of the coordinate system lies at the upper left corner of
        pixel [0, 0].

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
        raise NotImplementedError


DetectorResponse = np.ndarray


class Detector(Protocol):
    """Models a detector, such as a camera."""

    def response(self, photons: OpticsResponse, **kwargs) -> DetectorResponse:
        """Computes the response of the detector to an optical system."""
        raise NotImplementedError

    @property
    def num_pixels(self) -> Tuple[int, int]:
        """The number of pixels belonging to the detector."""
