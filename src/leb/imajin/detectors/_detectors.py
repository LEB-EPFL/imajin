from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import numpy as np
from numpy import random

from leb.imajin import Detector, DetectorResponse, OpticsResponse, Validation


class BitDepth(Enum):
    EIGHT = (8, np.uint8)
    TEN = (10, np.uint16)
    TWELVE = (12, np.uint16)
    SIXTEEN = (16, np.uint16)
    THIRTYTWO = (32, np.uint32)


@dataclass(frozen=True, slots=True)
class SimpleCMOSCamera(Detector, Validation):
    baseline: int = 100
    bit_depth: BitDepth = BitDepth.TWELVE
    dark_noise: float = 2.29
    num_pixels: Tuple[int, int] = (32, 32)
    quantum_efficiency: float = 0.69
    sensitivity: float = 5.88

    def validate_baseline(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("baseline must be greater than zero")
        return value

    def validate_dark_noise(self, value: float, **_) -> float:
        if value < 0:
            raise ValueError("dark_noise must be greater than zero")
        return value

    def validate_num_pixels(self, value: Tuple[int, int], **_) -> Tuple[int, int]:
        if value[0] < 0 or value[1] < 0:
            raise ValueError("num_pixels must be a 2-tuple of positive integers")
        return value

    def validate_quantum_efficiency(self, value: float, **_) -> float:
        if value < 0 or value > 1:
            raise ValueError("quantum_efficiency must be between 0 and 1, inclusive")
        return value

    def validate_sensitivity(self, value: float, **_) -> float:
        if value < 0:
            raise ValueError("sensitivity must be greater than zero")
        return value

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
