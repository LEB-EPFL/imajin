from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy import random

from leb.imajin import Detector, DetectorResponse, Optics, Sample, Source, Validation


@dataclass
class Simulator(Validation):
    detector: Detector
    optics: Optics
    sample: Sample
    source: Source

    time: float = 0
    dt: float = 0.01
    x_lim: Tuple[int, int] = (0, 32)
    y_lim: Tuple[int, int] = (0, 32)
    num_measurements: int = 100

    rng: Optional[random.Generator] = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = random.default_rng()

    def validate_num_measurements(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("num_measurements must be greater than zero")
        return value

    def step(self) -> DetectorResponse:
        sample_response = self.sample.response(self.time, self.dt, self.source)
        optics_response = self.optics.response(self.x_lim, self.y_lim, sample_response)
        detector_response = self.detector.response(optics_response, rng=self.rng)

        self.time += self.dt

        return detector_response

    def run(self) -> np.ndarray:
        rows, cols = self.detector.num_pixels
        measurements = np.zeros((self.num_measurements, rows, cols))
        for n in range(self.num_measurements):
            measurements[n, :, :] = self.step()

        return measurements
