from concurrent.futures import Executor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, List, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from leb.imajin import (
    Emitter,
    EmitterResponse,
    Sample,
    SampleResponse,
    Source,
    Validation,
)

from ._state_machine import Event, StateMachine

T = TypeVar("T", bound=npt.NBitBase)


class NullSample(Sample):
    def response(
        self, time: float, dt: float, source: Source, executor: Optional[Executor] = None
    ) -> SampleResponse:
        """A null sample does not respond to a radiation source."""
        return []


class ConstantEmitters(Sample):
    """A set of constant emitters of photons."""

    def __init__(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        z: npt.ArrayLike,
        rate: float,
        wavelength: float,
    ):
        self.x = np.asanyarray(x)
        self.y = np.asanyarray(y)
        self.z = np.asanyarray(z)
        self.rate = rate  # Number of photons emitted per unit of time
        self.wavelength = wavelength

    @property
    def rate(self) -> float:
        return self._photons

    @rate.setter
    def rate(self, value: float) -> None:
        if value < 0:
            raise ValueError("rate cannot be less than zero")

        self._photons = value

    @property
    def wavelength(self) -> float:
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value: float) -> None:
        if value <= 0:
            raise ValueError("wavelength must be greater than zero")

        self._wavelength = value

    def response(
        self, time: float, dt: float, source: Source, executor: Optional[Executor] = None
    ) -> SampleResponse:
        """Returns the response of the emitters to the radiation source.

        ConstantEmitters emit a constant number of photons per unit time interval, regardless of
        the state of the radiation source.

        The executor argument is ignored in this class; the response method is not parallelizable.

        """
        photons = int(self.rate * dt)
        return [
            EmitterResponse(x, y, z, photons, self.wavelength)
            for x, y, z in zip(self.x, self.y, self.z)
        ]


@dataclass
class Fluorophore(Emitter, Validation, Generic[T]):
    """A fluorophore with state transitions between fluorescent and non-fluorescent states.

    The number of photons emitted in the fluorescent state is the average number of photons
    emitted by an equivalent two-state system in which the transition from the excited to the
    ground state results in a photon.

    """

    x: np.floating[T]
    y: np.floating[T]
    z: np.floating[T]
    cross_section: float
    fluorescence_lifetime: float
    fluorescence_state: int
    quantum_yield: float
    state_machine: StateMachine
    wavelength: float

    def validate_cross_section(self, value: float, **_) -> float:
        if value <= 0:
            raise ValueError("cross_section must be greater than zero")
        return value

    def validate_fluorescence_lifetime(self, value: float, **_) -> float:
        if value <= 0:
            raise ValueError("fluorescence_lifetime must be greater than zero")
        return value

    def validate_fluorescence_state(self, value: int, **_) -> int:
        if value < 0:
            raise ValueError("fluorescence_state must be greater than or equal to zero")
        return value

    def validate_quantum_yield(self, value: float, **_) -> float:
        if value <= 0 or value > 1:
            raise ValueError("quantum_yield must be greater than zero and less than 1")
        return value

    def validate_wavelength(self, value: float, **_) -> float:
        if value <= 0:
            raise ValueError("wavelength must be greater than zero")
        return value

    def compute_on_fraction(self, time: float, dt: float, state_changes: List[Event]) -> float:
        """Returns a value between 0 and 1 representing the proportion of time in the ON state."""
        # No transitions occurred during the interval time + dt
        if len(state_changes) == 0 and self.fluorescence_state == self.state_machine.current_state:
            return 1
        if len(state_changes) == 0 and self.fluorescence_state != self.state_machine.current_state:
            return 0

        # Transitions occurred between energy levels during the interval time + dt
        # Break the interval time + dt into smaller intervals spent in different states
        intervals = []
        last_event_time = time
        for event in state_changes:
            intervals.append((event.time - last_event_time, event.from_state))
            last_event_time = event.time
        intervals.append((time + dt - last_event_time, state_changes[-1].to_state))

        on_fraction = sum(time for time, state in intervals if state == self.fluorescence_state) / (
            time + dt
        )
        return on_fraction

    def compute_photon_rate(self, irradiance: float) -> float:
        """Computes the number of fluorescent photons in response to an irradiance."""
        saturation_irradiance = (
            1 / self.cross_section / self.quantum_yield / self.fluorescence_lifetime
        )
        photons = (
            self.quantum_yield
            * self.cross_section
            * irradiance
            / (1 + irradiance / saturation_irradiance)
        )
        return photons

    def response(self, time: float, dt: float, source: Source) -> EmitterResponse:
        irradiance = np.array([source.irradiance(self.x, self.y)])
        state_changes = self.state_machine.collect(irradiance, time, dt)
        on_fraction = self.compute_on_fraction(time, dt, state_changes)
        photons = np.round(on_fraction * self.compute_photon_rate(irradiance[0]))

        return EmitterResponse(self.x, self.y, self.z, photons, self.wavelength)


def _parallel_response(func: Callable[[], EmitterResponse]) -> EmitterResponse:
    return func()


@dataclass
class Emitters(Sample):
    emitters: Sequence[Emitter]

    def response(
        self, time: float, dt: float, source: Source, executor: Optional[Executor] = None
    ) -> SampleResponse:
        if executor is None:
            return self._response_serial(time, dt, source)
        return self._response_parallel(time, dt, source, executor)

    def _response_serial(self, time: float, dt: float, source: Source) -> SampleResponse:
        responses = []
        for emitter in self.emitters:
            responses.append(emitter.response(time, dt, source))
        return responses

    def _response_parallel(
        self, time: float, dt: float, source: Source, executor: Executor
    ) -> SampleResponse:
        # Build a list of partial functions with zero arguments to workaround having to send args
        # to the executor
        funcs = [partial(emitter.response, time, dt, source) for emitter in self.emitters]
        return list(executor.map(_parallel_response, funcs))
