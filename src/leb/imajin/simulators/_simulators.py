from concurrent.futures import Executor
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, Protocol, Sequence, Tuple

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


@dataclass(frozen=True)
class StepResponse:
    """The data from a single simulation step.

    Attributes
    ----------
    sample_response: SampleResponse
        The response of the sample to the radiation source.
    optics_response: OpticsResponse
        The response of the optical system to the sample.
    detector_response: DetectorResponse
        The response of the detector to the optical system and sample.

    """

    sample_response: SampleResponse
    optics_response: OpticsResponse
    detector_response: DetectorResponse


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
        step_response: Optional[StepResponse] = None,
    ) -> None:
        """Processes the state of the simulation."""


def process(step: Callable[["Simulator"], StepResponse]) -> Callable[["Simulator"], StepResponse]:
    @wraps(step)
    def wrapper(self: "Simulator"):
        assert self.preprocessors is not None
        assert self.post_processors is not None

        for processor in self.preprocessors:
            processor(self)

        step_response = step(self)

        for processor in self.post_processors:
            processor(
                self,
                step_response=step_response,
            )

        return step_response

    return wrapper


@dataclass
class Simulator(Validation):
    """A Simulator integrates model components to generate simulated data.

    Parameters
    ----------
    detector: leb.imajin.Detector
    optics: leb.imajin.Optics
    sample: leb.imajin.Sample
    source: leb.imajin.Source
    time: float
    dt: float
    x_lim: tuple[int, int]
    y_lim: tuple[int, int]
    num_measurements: int
    preprocessors: leb.simulators.Processor
    postprocessors: leb.simulators.Processor
    executor: concurrent.futures.Executor
        If provided, an executor instance will be passed to the individual model components for
        parallel computation.
    rng: numpy.random.Generator
    backup: bool
        If True, a backup copy of the simulator will be made which will allow the Simulator
        instance to be reset.

    """

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

    executor: Optional[Executor] = None

    rng: Optional[random.Generator] = None

    backup: bool = True

    def __post_init__(self):
        if self.preprocessors is None:
            self.preprocessors = []

        if self.post_processors is None:
            self.post_processors = []

        if self.rng is None:
            self.rng = random.default_rng()

        if self.backup:
            self._initial_state = deepcopy(self)

    def __deepcopy__(self, memo) -> "Simulator":
        # Create a reference to the executor, instead of deepcopying it
        executor = self.executor

        return Simulator(
            deepcopy(self.detector, memo),
            deepcopy(self.optics, memo),
            deepcopy(self.sample, memo),
            deepcopy(self.source, memo),
            deepcopy(self.time, memo),
            deepcopy(self.dt, memo),
            deepcopy(self.x_lim, memo),
            deepcopy(self.y_lim, memo),
            deepcopy(self.num_measurements, memo),
            deepcopy(self.preprocessors, memo),
            deepcopy(self.post_processors, memo),
            executor=executor,
            rng=deepcopy(self.rng, memo),
            backup=False,
        )

    def validate_num_measurements(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("num_measurements must be greater than zero")
        return value

    @process
    def step(self) -> StepResponse:
        sample_response = self.sample.response(self.time, self.dt, self.source, self.executor)
        optics_response = self.optics.response(self.x_lim, self.y_lim, sample_response)
        detector_response = self.detector.response(optics_response, rng=self.rng)

        self.time += self.dt

        return StepResponse(sample_response, optics_response, detector_response)

    def run(self, reset: bool = False) -> np.ndarray:
        rows, cols = self.detector.num_pixels
        measurements = np.zeros((self.num_measurements, rows, cols))
        for num in range(self.num_measurements):
            step_response = self.step()
            measurements[num, :, :] = step_response.detector_response

        if reset:
            self.reset()

        return measurements

    def reset(self) -> None:
        """Resets the simulator to its original state."""
        if self.backup:
            self.__dict__.update(self._initial_state.__dict__)
            self.backup = True
        else:
            raise RuntimeError("The simulator instance cannot be reset.")
