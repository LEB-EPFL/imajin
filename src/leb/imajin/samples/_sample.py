from dataclasses import dataclass
from typing import Generic, List, Optional, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from leb.imajin import Validation
from leb.imajin.instruments import Source

T1 = TypeVar("T1", bound=npt.NBitBase)


@dataclass(frozen=True)
class EmitterResponse(Validation, Generic[T1]):
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


class Sample(Protocol):
    def response(self, source: Source, dt: float) -> Optional[List[EmitterResponse]]:
        pass
