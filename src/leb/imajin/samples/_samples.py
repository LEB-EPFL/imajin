from dataclasses import dataclass
from typing import Generic, List, TypeVar

import numpy as np
import numpy.typing as npt

from leb.imajin import EmitterResponse, Sample, SampleResponse, Source


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

    def response(self, source: Source, dt: float) -> SampleResponse:
        """Returns the response of the emitters to the radiation source.

        ConstantEmitters emit a constant number of photons per unit time interval, regardless of
        the state of the radiation source.

        """
        photons = int(self.rate * dt)
        return [
            EmitterResponse(x, y, z, photons, self.wavelength)
            for x, y, z in zip(self.x, self.y, self.z)
        ]
