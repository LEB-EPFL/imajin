from dataclasses import dataclass
from typing import Tuple

import numpy as np

from leb.imajin.samples import SampleResponse

from ._optical_system import PSF, OpticalSystem


@dataclass
class SimpleMicroscope(OpticalSystem):
    psf: PSF

    def response(
        self,
        x_lim: Tuple[int, int],
        y_lim: Tuple[int, int],
        sample_response: SampleResponse,
    ) -> np.ndarray:
        self._validate(x_lim, y_lim)
        photons = np.zeros((y_lim[1], x_lim[1]))
        y, x = np.ogrid[y_lim[0] : y_lim[1], x_lim[0] : x_lim[1]]

        for emitter in sample_response:
            photons += np.uint(
                self.psf.bin(x, y, emitter.x, emitter.y) * emitter.photons
            )

        return photons

    def _validate(self, x_lim: Tuple[int, int], y_lim: Tuple[int, int]):
        if x_lim[0] >= x_lim[1]:
            raise ValueError("The first element of x_lim must be less than the second")
        if y_lim[0] >= y_lim[1]:
            raise ValueError("The first element of y_lim must be less than the second")
