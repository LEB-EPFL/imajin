import os
from dataclasses import InitVar, dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple

import numpy as np

CACHE_SIZE_SM_RATES: int = int(os.environ.get("IMAJIN_CACHE_SIZE_SM_RATES", 100000))
CACHE_SIZE_SM_STOPPED_STATES: int = int(os.environ.get("IMAJIN_CACHE_SIZE_SM_STOPPED_STATES", 1))


@dataclass
class Event:
    """A state machine event."""

    time: float
    from_state: int
    to_state: int


def to_tuple(array):
    """Converts any numpy array to (possibly) nested tuples."""
    try:
        return tuple(to_tuple(i) for i in array)
    except TypeError:
        return array


@lru_cache(maxsize=CACHE_SIZE_SM_RATES)
def compute_rates_cached(
    control_params: Tuple[float, ...],
    rate_constants: Tuple[Tuple[float]],
    rate_coefficients: Tuple[Tuple[Tuple[Tuple[float]]]],
) -> np.ndarray:
    _control_params: np.ndarray = np.array(control_params)
    _rate_constants: np.ndarray = np.array(rate_constants)
    _rate_coefficients: np.ndarray = np.array(rate_coefficients)

    if _rate_coefficients.shape == (0,):
        # Rates do not depend on any control parameters
        return np.array(_rate_constants)
    if _control_params.ndim != 1:
        raise ValueError("the control parameters array must have a dimension of 1")
    if _control_params.shape[0] != _rate_coefficients.shape[0]:
        raise ValueError(
            "the control parameters array must have the same number of elements as the first "
            "dimension of the rate_coefficents array"
        )

    # Creates an L x M array where each row is a control parameter and each column is a control
    # parameter value raised to a power equal to the column number starting from 1. For example,
    # if control_params is np.array([1, 2]) and M = 3, then powers is
    # np.array([[1, 1, 1], [2, 4, 8]]).
    powers = np.power(
        _control_params[:, np.newaxis],
        np.arange(1, _rate_coefficients.shape[1] + 1),
    )

    # Returns a N x N array
    return _rate_constants + np.tensordot(powers, _rate_coefficients)


@lru_cache(maxsize=CACHE_SIZE_SM_STOPPED_STATES)
def stopped_states_cached(
    rate_constants: Tuple[Tuple[float]], rate_coefficients: Tuple[Tuple[Tuple[Tuple[float]]]]
) -> List[int]:
    """Caches the results of StateMachine.stopped_states()"""
    _rate_constants: np.ndarray = np.array(rate_constants)
    _rate_coefficients: np.ndarray = np.array(rate_coefficients)

    num_states = _rate_constants.shape[0]
    stopped_states_indexes = [True] * num_states

    for control_param in _rate_coefficients:
        for power in control_param:
            for from_state, rates in enumerate(power):
                if any(rates != 0):
                    stopped_states_indexes[from_state] = False

    for from_state, constants in enumerate(_rate_constants):
        if any(constants != 0):
            stopped_states_indexes[from_state] = False

    stopped_states = [x for x in range(num_states) if stopped_states_indexes[x]]
    return stopped_states


@dataclass
class StateMachine:
    """A state machine with transitions between states that are exponential random processes.

    Transition rate constants are modeled by an N x N array where each row indicates the state
    being transitioned from and each column indicates the state being transitioned to.

    The transition rates between states may also be functions of control parameters. The
    functional relationship between a rate and a control parameter p is modeled as a power series
    expansion in p. If there are L control parameters, M is the highest order of the expansion
    that is retained, and there are N states, then there is a L x M x N x N array that contains
    the information about the dependence of the rates on the control parameters.

    For example, rate_coefficients[0, 1, 3, 2] contains the second order power series
    expansion coefficient of the rate coefficient's dependence on control parameter 0 for the
    transition from state 3 to state 2. (For the M'th dimension, index 0 corresponds to the first
    order, index 1 to the second order, etc. Zero order coefficients are in the rate_constants
    matrix.)

    If the state machine has rate constants only, i.e. rates that do not change with any control
    parameter, then use empty arrays (np.array([])) for `rate_coefficients` and `control_params`.

    Attributes
    ----------
    current_state : int
        An integer that identifies the machine's current state. For a machine with N states, this
        value must be between 0 and N - 1.
    rate_constants: numpy.ndarray
        An N x N array containing the transition rates that are independent of the control
        parameters.
    rate_coefficients: numpy.ndarray
        An L x M x N x N array containing the power series expansion coefficients for the
        functional dependence of the rates on the L control parameters. The value of M indicates
        the highest order of the control parameters that is retained in the expansion.
    stopped: bool
        Whether the state machine can no longer advance out of its current state.
    rng: numpy.random.Generator
        A random number generator

    """

    current_state: int
    control_params: InitVar[np.ndarray]
    _control_params: np.ndarray = field(init=False, repr=False)
    _next_event: Event = field(init=False, repr=False)

    rate_constants: np.ndarray
    rate_coefficients: np.ndarray

    # Used for caching
    _rate_constants: Tuple[Tuple[float]] = field(init=False, repr=False)
    _rate_coefficients: Tuple[Tuple[Tuple[Tuple[float]]]] = field(init=False, repr=False)

    stopped: bool = False
    rng: Optional[np.random.Generator] = field(default=None, repr=False)

    def __post_init__(self, control_params):
        if self.rng is None:
            self.rng = np.random.default_rng()

        if control_params.ndim != 1:
            raise ValueError("the control parameters array must have a dimension of 1")
        if control_params.shape[0] != self.rate_coefficients.shape[0]:
            raise ValueError(
                "the control parameters array must have the same number of elements as the first "
                "dimension of the rate_coefficents array"
            )

        self._control_params = control_params
        self._rate_constants = to_tuple(self.rate_constants)
        self._rate_coefficients = to_tuple(self.rate_coefficients)
        self._next_event = self._compute_next_event(control_params, 0)

    def collect(self, control_params: np.ndarray, time: float, dt: float) -> List[Event]:
        """Step a state machine and collect its transition events over a time period.

        This is the sole public interface to a StateMachine.

        """
        if self.stopped:
            return []

        self._update(control_params, time)

        events = []
        while self._next_event.time < time + dt:
            events.append(self._next_event)
            self._step(control_params, self._next_event.time)

            if self.stopped:
                break

        return events

    def _step(self, control_params: np.ndarray, t_offset: float) -> None:
        """Steps the state machine to the next state and computes the event that follows.

        t_offset is used to inject a master clock by adding an offset to the time of the next
        event that is generated.

        """
        self.current_state = self._next_event.to_state
        self._next_event = self._compute_next_event(control_params, t_offset)

    def _compute_rates(self, control_params: np.ndarray) -> np.ndarray:
        return compute_rates_cached(
            tuple(control_params), self._rate_constants, self._rate_coefficients
        )

    def _compute_next_event(self, control_params: np.ndarray, t_offset: float) -> Event:
        if self.current_state in self._stopped_states():
            # All rates are zero, so the state machine can no longer advance.
            self.stopped = True

            return Event(time=np.inf, from_state=self.current_state, to_state=self.current_state)

        # Find all transition rates from the current state
        rates = self._compute_rates(control_params)[self.current_state, :]

        # Take N samples from exponential distributions with scale factors equal to the reciprocal
        # of the values in rates
        assert isinstance(self.rng, np.random.Generator)
        transition_times = self.rng.exponential(
            np.reciprocal(rates, out=np.full(rates.shape, np.inf), where=rates > 0)
        )

        return Event(
            time=np.min(transition_times) + t_offset,
            from_state=self.current_state,
            to_state=np.argmin(transition_times).astype(int),
        )

    def _update(self, control_params: np.ndarray, t_offset: float) -> None:
        """Updates the state machine's next event without advancing its state.

        This is to be used to handle the case when the values of the control parameters change.

        """
        if np.array_equal(control_params, self._control_params):
            return
        self._next_event = self._compute_next_event(control_params, t_offset)

    def _stopped_states(self) -> List[int]:
        """Finds stopped states of the state machine.

        Stopped states are states with all zero rate constants and coefficients to other states.

        Returns
        -------
        List[int]
            Integer IDs of states from which the machine cannot advance.

        """
        return stopped_states_cached(self._rate_constants, self._rate_coefficients)
