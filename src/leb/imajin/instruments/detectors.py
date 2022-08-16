from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from numpy.random import RandomState


class BitDepth(Enum):
    EIGHT = (8, np.uint8)
    TEN = (10, np.uint16)
    TWELVE = (12, np.uint16)
    SIXTEEN = (16, np.uint16)
    THIRTYTWO = (32, np.uint32)


class SimpleCMOSCamera:
    def __init__(
        self,
        baseline: int = 100,
        bit_depth: BitDepth = BitDepth.TWELVE,
        dark_noise: float = 2.29,
        num_pixels: Tuple[int, int] = (32, 32),
        qe: float = 0.69,
        sensitivity: float = 5.88,
    ):
        self.baseline = baseline
        self.bit_depth = bit_depth
        self.dark_noise = dark_noise
        self.num_pixels = num_pixels
        self.qe = qe
        self.sensitivity = sensitivity

    def __repr__(self) -> str:
        return f"SimpleCMOSCamera(baseline={self.baseline}, bit_depth={self.bit_depth}, dark_noise={self.dark_noise}, qe={self.qe}, sensitivity={self.sensitivity})"

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
    def qe(self) -> float:
        return self._qe

    @qe.setter
    def qe(self, value: float) -> None:
        if value < 0 or value > 1:
            raise ValueError("qe must be between 0 and 1, inclusive")

        self._qe = value

    @property
    def sensitivity(self) -> float:
        return self._sensitivity

    @sensitivity.setter
    def sensitivity(self, value: float) -> None:
        if value < 0:
            raise ValueError("sensitivity must be greater than zero")

        self._sensitivity = value

    def get_image(
        self,
        photons: Optional[npt.NDArray[np.integer]] = None,
        rs: Optional[RandomState] = None,
    ) -> npt.NDArray[np.unsignedinteger]:
        if rs is None:
            rs = RandomState()

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
            photoelectrons = rs.poisson(self.qe * photons, size=photons.shape)  # type: ignore

        # Add dark noise
        electrons = (
            rs.normal(scale=self.dark_noise, size=photoelectrons.shape) + photoelectrons
        )

        # Convert to ADU and add baseline
        adu = electrons * self.sensitivity
        adu += self.baseline

        # Model pixel saturation
        bits, data_type = self.bit_depth.value
        max_adu = data_type(2**bits - 1)
        adu[adu > max_adu] = max_adu

        return adu.astype(data_type)
