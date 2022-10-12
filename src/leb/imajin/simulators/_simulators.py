from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, Tuple

import numpy as np
from numpy import random

from leb.imajin import (
    Detector,
    DetectorResponse,
    Optics,
    OpticsResponse,
    Sample,
    SampleResponse,
    Source,
    Validation,
)


class Processor(Protocol):
    """A class for processing or modifying the simulation state at each time step.

    A processor is a Callable that is called before and after each simulation step.

    This may be useful for constructing reward functions, writing the state of the simulation to a
    file, or modifying a simulation component value in response to the state, for example. Any
    results should be stored within the Processor instance because the Callable does not return a
    value.

    """

    def __call__(
        self,
        simulator: "Simulator",
        detector_response: Optional[DetectorResponse] = None,
        sample_response: Optional[SampleResponse] = None,
        optics_response: Optional[OpticsResponse] = None,
    ) -> None:
        """Processes the state of the simulation."""


@dataclass
class Simulator(Validation):
    detector: Detector
    optics: Optics
    sample: Sample
    source: Source

    time: float = 0
    dt: float = 1.0
    x_lim: Tuple[int, int] = (0, 32)
    y_lim: Tuple[int, int] = (0, 32)
    num_measurements: int = 100

    preprocessors: Optional[Sequence[Processor]] = None
    post_processors: Optional[Sequence[Processor]] = None

    rng: Optional[random.Generator] = None

    def __post_init__(self):
        if self.preprocessors is None:
            self.preprocessors = []

        if self.post_processors is None:
            self.post_processors = []

        if self.rng is None:
            self.rng = random.default_rng()

        self._initial_state = deepcopy(self)

    def validate_num_measurements(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("num_measurements must be greater than zero")
        return value

    def step(self) -> DetectorResponse:
        assert self.preprocessors is not None
        assert self.post_processors is not None

        for processor in self.preprocessors:
            processor(self)

        sample_response = self.sample.response(self.time, self.dt, self.source)
        optics_response = self.optics.response(self.x_lim, self.y_lim, sample_response)
        detector_response = self.detector.response(optics_response, rng=self.rng)

        self.time += self.dt

        for processor in self.post_processors:
            processor(
                self,
                detector_response=detector_response,
                sample_response=sample_response,
                optics_response=optics_response,
            )

        return detector_response

    def run(self, reset: bool = False) -> np.ndarray:
        rows, cols = self.detector.num_pixels
        measurements = np.zeros((self.num_measurements, rows, cols))
        for num in range(self.num_measurements):
            measurements[num, :, :] = self.step()

        if reset:
            self.reset()

        return measurements

    def reset(self) -> None:
        """Resets the simulator to its original state."""
        self.__dict__.update(self._initial_state.__dict__)
