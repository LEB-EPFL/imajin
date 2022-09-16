from dataclasses import dataclass
from unittest.mock import create_autospec

import numpy as np
import numpy.typing as npt
import pytest

from leb.imajin.samples import Fluorophore
from leb.imajin.samples._state_machine import Event, StateMachine


@pytest.fixture
def state_machine(rate_constants):
    """A 2x2x2x2 state machine with a fake random number generator."""
    rng = create_autospec(np.random.Generator, spec_set=True)
    s = StateMachine(
        0,
        np.array([1, 1]),
        rate_constants,
        np.array(
            [
                [[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
                [[[0, 1], [1, 0]], [[0, 1], [1, 0]]],
            ]
        ),
        rng=rng,
    )
    # Set the first event because the RNG is a fake
    s._next_event = Event(time=0, from_state=0, to_state=1)
    return s


@pytest.fixture
def stopping_state_machine():
    """A 2x2x2x2 state machine that stops once it reaches state 1.

    Stopping is encoded as rows that are all zeros in the 2x2 rate matrices.

    """
    rng = create_autospec(np.random.Generator, spec_set=True)
    s = StateMachine(
        0,
        np.array([1, 1]),
        np.array([[0, 1], [0, 0]]),
        np.array(
            [
                [[[0, 1], [0, 0]], [[0, 1], [0, 0]]],
                [[[0, 1], [0, 0]], [[0, 1], [0, 0]]],
            ]
        ),
        rng=rng,
    )
    # Set the first event because the RNG is a fake
    s._next_event = Event(time=0, from_state=0, to_state=1)
    return s


@pytest.fixture
def fluorophore():
    """A fluorophore with a dummy state machine."""
    state_machine = create_autospec(StateMachine)
    return Fluorophore(
        x=np.float_(0),
        y=np.float_(0),
        z=np.float_(0),
        cross_section=1e-6,
        fluorescence_lifetime=1e-6,
        fluorescence_state=0,
        quantum_yield=0.8,
        state_machine=state_machine,
        wavelength=7,
    )


@dataclass
class RatesTestData:
    control_params: npt.ArrayLike
    rate_constants: npt.ArrayLike
    rate_coefficients: npt.ArrayLike
    expected_result: npt.ArrayLike


@pytest.fixture
def rate_constants():
    """Rate constants for a N = 2 state machine."""
    return np.array([[0, 1], [0.5, 0]])


@pytest.fixture
def two_control_params_second_order(rate_constants):
    """Rate coefficients for state machine with two control parameters and second order expansion.

    Constant rate coefficients shape is 2 x 2.

    Coefficient matrix shape is 2 x 2 x 2 x 2.

    Result shape is 2 x 2.

    """
    return RatesTestData(
        control_params=np.array([2, 3]),
        rate_constants=rate_constants,
        rate_coefficients=np.array(
            [
                [[[0, 0], [2, 0]], [[0, 2], [0.5, 0]]],
                [[[0, 1], [4, 0]], [[0, 2], [0.5, 0]]],
            ]
        ),
        expected_result=np.array([[0, 30], [23, 0]]),
    )


@pytest.fixture
def two_control_params_third_order(rate_constants):
    """Rate coefficients for state machine with two control parameters and third order expansion.

    Constant rate coefficients shape is 2 x 2.

    Coefficient matrix shape is 2 x 3 x 2 x 2.

    Result shape is 2 x 2.

    """
    return RatesTestData(
        control_params=np.array([2, 3]),
        rate_constants=rate_constants,
        rate_coefficients=np.array(
            [
                [[[0, 0], [2, 0]], [[0, 2], [0.5, 0]], [[0, 1], [1, 0]]],
                [[[0, 1], [4, 0]], [[0, 2], [0.5, 0]], [[0, 2], [2, 0]]],
            ]
        ),
        expected_result=np.array([[0, 92], [85, 0]]),
    )


@pytest.fixture
def three_control_params_second_order(rate_constants):
    """Rate coefficients for state machine with three control parameters and second order expansion.

    Constant rate coefficients shape is 2 x 2.

    Coefficient matrix shape is 2 x 3 x 2 x 2.

    Result shape is 2 x 2.

    """
    return RatesTestData(
        control_params=np.array([2, 3, 2]),
        rate_constants=rate_constants,
        rate_coefficients=np.array(
            [
                [[[0, 0], [2, 0]], [[0, 2], [0.5, 0]]],
                [[[0, 1], [4, 0]], [[0, 2], [0.5, 0]]],
                [[[0, 1], [1, 0]], [[0, 2], [2, 0]]],
            ]
        ),
        expected_result=np.array([[0, 40], [33, 0]]),
    )


@pytest.fixture
def zero_control_params(rate_constants):
    """Rate coefficients for state machine with only constant rate coefficients.

    Constant rate coefficients shape is 2 x 2.

    Coefficient matrix shape is (0,)

    Result shape is 2 x 2.

    """
    return RatesTestData(
        control_params=np.array([]),
        rate_constants=rate_constants,
        rate_coefficients=np.array([]),
        expected_result=rate_constants,
    )
