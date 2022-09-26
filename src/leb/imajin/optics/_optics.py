from dataclasses import dataclass
from typing import Tuple

import numpy as np
import numpy.typing as npt

from leb.imajin import PSF, Optics, SampleResponse


def safe_round(array: npt.ArrayLike, total: int) -> np.ndarray:
    """Rounds an array of floats, maintaining their integer sum."""
    array = np.asanyarray(array)

    # Round the array to the nearest integer
    rounded_array: np.ndarray = np.rint(array)
    error = total - np.sum(rounded_array)

    if error == 0:
        return rounded_array

    # The number of elements to adjust. For integers, each element after rounding is within 0.5 of
    # the desired value, so the maximum adjustment is 1.
    num_elements_to_adjust = int(np.abs(error))

    # np.argsort() returns an array of indices that would sort an array
    sorted_index_array = np.argsort(array - rounded_array, axis=None)

    # Add +/- 1 to the elements of the rounded_array with the n largest rounding errors
    safe_rounded_array = rounded_array.flatten()
    safe_rounded_array[sorted_index_array[0:num_elements_to_adjust]] += np.copysign(1, error)

    return safe_rounded_array.reshape(array.shape)


@dataclass
class SimpleMicroscope(Optics):
    psf: PSF

    def response(
        self,
        x_lim: Tuple[int, int],
        y_lim: Tuple[int, int],
        sample_response: SampleResponse,
    ) -> np.ndarray:
        self._validate(x_lim, y_lim)
        total_photons = np.zeros((y_lim[1], x_lim[1]))
        y, x = np.ogrid[y_lim[0] : y_lim[1], x_lim[0] : x_lim[1]]

        # TODO Handle truncation of an emitter by the edges of the computational grid
        for emitter in sample_response:
            emitted_photons = self.psf.bin(x, y, emitter.x, emitter.y) * emitter.photons
            emitted_photons = safe_round(emitted_photons, emitter.photons)
            total_photons += emitted_photons

        return total_photons.astype(np.uint64)

    def _validate(self, x_lim: Tuple[int, int], y_lim: Tuple[int, int]):
        if x_lim[0] >= x_lim[1]:
            raise ValueError("The first element of x_lim must be less than the second")
        if y_lim[0] >= y_lim[1]:
            raise ValueError("The first element of y_lim must be less than the second")
