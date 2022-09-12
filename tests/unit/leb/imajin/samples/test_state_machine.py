from ossaudiodev import control_labels
from unittest.mock import create_autospec

import numpy as np
import pytest

from leb.imajin.samples._state_machine import Event, StateMachine


class TestAdvanceMachineState:
    @pytest.mark.usefixtures("state_machine")
    def test_step(self, state_machine):
        # Get the current control parameters with which the machine was initialized
        control_params = state_machine._control_params
        # RNG returns its inputs unmodified so that transition times are the reciprocal rates
        state_machine.rng.exponential = lambda x: x
        rates = state_machine._compute_rates(control_params)
        times = np.reciprocal(rates, out=np.full(rates.shape, np.inf), where=rates > 0)

        assert state_machine.current_state == 0

        state_machine._step(control_params, t_offset=42)

        assert state_machine.current_state == 1
        assert state_machine._next_event.time == np.min(times[1, :]) + 42
        assert state_machine._next_event.from_state == 1
        assert state_machine._next_event.to_state == 0

        state_machine._step(control_params, t_offset=42)

        assert state_machine.current_state == 0
        assert state_machine._next_event.time == np.min(times[0, :]) + 42
        assert state_machine._next_event.from_state == 0
        assert state_machine._next_event.to_state == 1

    @pytest.mark.usefixtures("state_machine")
    def test_collect(self, state_machine):
        # Get the current control parameters with which the machine was initialized
        control_params = state_machine._control_params
        # RNG returns its inputs unmodified so that transition times are the reciprocal rates
        state_machine.rng.exponential = lambda x: x
        rates = state_machine._compute_rates(control_params)
        times = np.reciprocal(rates, out=np.full(rates.shape, np.inf), where=rates > 0)

        assert state_machine.current_state == 0

        events = state_machine.collect(control_params, 0, 1)

        # 4 events happen between time 0 and 1, plus the first event at t=0
        # t_01 = 0.2, t_10 = 0.2222
        assert 5 == len(events)

        expected_times = [0, 0.2222, 0.4222, 0.6444, 0.8444]
        for time, event in zip(expected_times, events):
            np.testing.assert_almost_equal(time, event.time, decimal=4)

    @pytest.mark.usefixtures("state_machine")
    def test_no_events_when_machine_is_stopped(self, state_machine):
        control_params = np.array([1, 1])
        state_machine.stopped = True

        events = state_machine.collect(control_params, 0, 999999999)

        assert len(events) == 0

    @pytest.mark.usefixtures("stopping_state_machine")
    def test_machine_cannot_move_past_stopped_state(self, stopping_state_machine):
        # Get the current control parameters with which the machine was initialized
        control_params = stopping_state_machine._control_params
        # RNG returns its inputs unmodified so that transition times are the reciprocal rates
        stopping_state_machine.rng.exponential = lambda x: x

        assert stopping_state_machine.current_state == 0
        assert stopping_state_machine.stopped is False

        events = stopping_state_machine.collect(control_params, 0, 1)

        assert len(events) == 1
        assert stopping_state_machine.current_state == 1
        assert stopping_state_machine.stopped

        events = stopping_state_machine.collect(control_params, 1, 99999999)

        assert len(events) == 0
        assert stopping_state_machine.current_state == 1
        assert stopping_state_machine.stopped


class TestNextEvent:
    @pytest.mark.usefixtures("state_machine")
    def test_compute_next_event(self, state_machine):
        # Transition time to state 0 is infinite, to state 1 is 1
        state_machine.rng.exponential.return_value = np.array([np.inf, 1])

        result = state_machine._compute_next_event(np.array([2, 4]), t_offset=42)

        # Event occurs at time 1 + t_offset
        assert result == Event(43, state_machine.current_state, 1)


class TestComputeRates:
    @pytest.mark.usefixtures("two_control_params_second_order")
    def test_two_control_params_second_order(self, two_control_params_second_order):
        s = StateMachine(
            0,
            two_control_params_second_order.control_params,
            two_control_params_second_order.rate_constants,
            two_control_params_second_order.rate_coefficients,
        )

        result = s._compute_rates(two_control_params_second_order.control_params)

        np.testing.assert_array_equal(
            two_control_params_second_order.expected_result, result
        )

    @pytest.mark.usefixtures("two_control_params_third_order")
    def test_two_control_params_third_order(self, two_control_params_third_order):
        s = StateMachine(
            0,
            two_control_params_third_order.control_params,
            two_control_params_third_order.rate_constants,
            two_control_params_third_order.rate_coefficients,
        )

        result = s._compute_rates(two_control_params_third_order.control_params)

        np.testing.assert_array_equal(
            two_control_params_third_order.expected_result, result
        )

    @pytest.mark.usefixtures("three_control_params_second_order")
    def test_three_control_params_second_order(self, three_control_params_second_order):
        s = StateMachine(
            0,
            three_control_params_second_order.control_params,
            three_control_params_second_order.rate_constants,
            three_control_params_second_order.rate_coefficients,
        )

        result = s._compute_rates(three_control_params_second_order.control_params)

        np.testing.assert_array_equal(
            three_control_params_second_order.expected_result, result
        )

    @pytest.mark.usefixtures("zero_control_params")
    def test_zero_control_params(self, zero_control_params):
        s = StateMachine(
            0,
            zero_control_params.control_params,
            zero_control_params.rate_constants,
            zero_control_params.rate_coefficients,
        )

        result = s._compute_rates(zero_control_params.control_params)

        np.testing.assert_array_equal(zero_control_params.expected_result, result)
