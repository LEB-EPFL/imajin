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


class CMOSCamera:
    def __init__(
        self,
        qe: float = 0.69,
        sensitivity: float = 5.88,
        dark_noise: float = 2.29,
        bit_depth: BitDepth = BitDepth.TWELVE,
        baseline: int = 100,
    ):
        if qe < 0 or qe > 1:
            raise ValueError("qe must be between 0 and 1, inclusive")

        if sensitivity < 0:
            raise ValueError("sensitivity must be greater than zero")

        if dark_noise < 0:
            raise ValueError("dark_noise must be greater than zero")

        if baseline < 0:
            raise ValueError("baseline must be greater than zero")

        self.qe = qe
        self.sensitivity = sensitivity
        self.dark_noise = dark_noise
        self.bit_depth = bit_depth
        self.baseline = baseline

    def snapshot(
        self,
        photons: npt.NDArray[np.unsignedinteger],
        rs: Optional[RandomState]= None,
    ) -> npt.NDArray[np.unsignedinteger]:
        if rs is None:
            rs = RandomState()

        # Add shot noise and convert to electrons
        photoelectrons = rs.poisson(self.qe * photons, size=photons.shape)

        # Add dark noise
        electrons = rs.normal(scale=self.dark_noise, size=photoelectrons.shape) + photoelectrons

        # Convert to ADU and add baseline
        bits, data_type = self.bit_depth.value
        max_adu = data_type(2**bits - 1)
        adu = (electrons * self.sensitivity).astype(data_type) # Convert to discrete numbers
        adu += self.baseline
        adu[adu > max_adu] = max_adu  # models pixel saturation

        return adu
