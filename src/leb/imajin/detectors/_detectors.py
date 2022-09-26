from enum import Enum
from typing import Tuple

import numpy as np
from numpy import random

from leb.imajin import Detector, DetectorResponse, OpticsResponse


class BitDepth(Enum):
    EIGHT = (8, np.uint8)
    TEN = (10, np.uint16)
    TWELVE = (12, np.uint16)
    SIXTEEN = (16, np.uint16)
    THIRTYTWO = (32, np.uint32)


class SimpleCMOSCamera(Detector):
    def __init__(
        self,
        baseline: int = 100,
        bit_depth: BitDepth = BitDepth.TWELVE,
        dark_noise: float = 2.29,
        num_pixels: Tuple[int, int] = (32, 32),
        quantum_efficiency: float = 0.69,
        sensitivity: float = 5.88,
    ):
        self.baseline = baseline
        self.bit_depth = bit_depth
        self.dark_noise = dark_noise
        self.num_pixels = num_pixels
        self.quantum_efficiency = quantum_efficiency
        self.sensitivity = sensitivity

    def __repr__(self) -> str:
        return (
            f"SimpleCMOSCamera(baseline={self.baseline}, "
            f"bit_depth={self.bit_depth}, dark_noise={self.dark_noise}, "
            f"num_pixels={self.num_pixels}, quantum_efficiency={self.quantum_efficiency}, "
            f"sensitivity={self.sensitivity})"
        )

    @property
    def baseline(self) -> int:
        return self._baseline

    @baseline.setter
    def baseline(self, value: int) -> None:
        if value < 0:
            raise ValueError("baseline must be greater than zero")

        self._baseline = value

    @property
    def dark_noise(self) -> float:
        return self._dark_noise

    @dark_noise.setter
    def dark_noise(self, value: float) -> None:
        if value < 0:
            raise ValueError("dark_noise must be greater than zero")

        self._dark_noise = value

    @property
    def num_pixels(self) -> Tuple[int, int]:
        return self._num_pixels

    @num_pixels.setter
    def num_pixels(self, value: Tuple[int, int]) -> None:
        if value[0] < 0 or value[1] < 0:
            raise ValueError("num_pixels must be a 2-tuple of positive integers")

        self._num_pixels = value

    @property
    def quantum_efficiency(self) -> float:
        return self._quantum_efficiency

    @quantum_efficiency.setter
    def quantum_efficiency(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("quantum_efficiency must be between 0 and 1, inclusive")

        self._quantum_efficiency = value

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        if value < 0:
            raise ValueError("sensitivity must be greater than zero")

        self._sensitivity = value

    def response(self, photons: OpticsResponse, **kwargs) -> DetectorResponse:
        """Computes the response of the detector to an optical system."""
        if (rng := kwargs.get("rng")) is None:
            rng = random.default_rng()

        if photons is None:
            # No signal
            photoelectrons = np.zeros(self.num_pixels)
        elif photons.shape != self.num_pixels:
            raise ValueError("photons must have the same shape as num_pixels")
        elif np.any(photons[photons < 0]):
            raise ValueError("photons cannot be less than zero")
        else:
            # Add shot noise and convert to electrons
            # Ignore typing so numpy can handle intermediate data types without mypy errors
            photoelectrons = rng.poisson(
                self.quantum_efficiency * photons, size=photons.shape
            )  # type: ignore

        # Add dark noise
        electrons = rng.normal(scale=self.dark_noise, size=photoelectrons.shape) + photoelectrons

        # Convert to ADU and add baseline
        adu = electrons * self.sensitivity
        adu += self.baseline

        # Model pixel saturation
        bits, data_type = self.bit_depth.value
        max_adu = data_type(2**bits - 1)
        adu[adu > max_adu] = max_adu

        return adu.astype(data_type)
