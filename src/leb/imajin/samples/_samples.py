from dataclasses import dataclass
from typing import Generic, Iterable, TypeVar

import numpy as np
import numpy.typing as npt

from leb.imajin import Response, Validation
from leb.imajin.instruments import Source

from ._sample import Sample

T1 = TypeVar("T1", bound=npt.NBitBase)


@dataclass(frozen=True)
class EmitterToSource(Validation, Generic[T1]):
    """Models the response of single fluorescent emitters to a radiation source."""

    x: np.floating[T1]
    y: np.floating[T1]
    z: np.floating[T1]
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


@dataclass(frozen=True)
class EmittersToSource(Response):
    """Models the response of a set of fluorescent emitters to a radiation source."""

    emitters: Iterable[EmitterToSource]

    def __post_init__(self):
        object.__setattr__(self, "emitters", list(self.emitters))

    def __iter__(self):
        return iter(self.emitters)

    def __len__(self):
        return len(self.emitters)


class NullSample(Sample):
    def response(self, source: Source, dt: float) -> None:
        """A null sample does not respond to a radiation source."""


class ConstantEmitters(Sample):
    """A set of constant emitters of photons."""

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        z: npt.ArrayLike,
        rate: float,
        wavelength: float,
    ):
        self.x = np.asanyarray(x)
        self.y = np.asanyarray(y)
        self.z = np.asanyarray(z)
        self.rate = rate  # Number of photons emitted per unit of time
        self.wavelength = wavelength

    @property
    def rate(self) -> float:
        return self._photons

    @rate.setter
    def rate(self, value: float) -> None:
        if value < 0:
            raise ValueError("rate cannot be less than zero")

        self._photons = value

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        if value <= 0:
            raise ValueError("wavelength must be greater than zero")

        self._wavelength = value

    def response(self, source: Source, dt: float) -> EmittersToSource:
        """Returns the response of the emitters to the radiation source.

        ConstantEmitters emit a constant number of photons per unit time interval, regardless of
        the state of the radiation source.

        """
        photons = int(self.rate * dt)
        return EmittersToSource(
            EmitterToSource(x, y, z, photons, self.wavelength)
            for x, y, z in zip(self.x, self.y, self.z)
        )
