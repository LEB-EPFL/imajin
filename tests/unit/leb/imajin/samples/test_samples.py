from typing import Iterable
from unittest.mock import create_autospec

import numpy as np
import pytest

from leb.imajin import Source
from leb.imajin.samples import *


class TestEmitterResponse:
    @pytest.mark.parametrize(
        "inputs",
        [
            {"photons": -100, "wavelength": 532.0},
            {"photons": 100, "wavelength": -532.0},
            {"photons": -100, "wavelength": -532.0},
        ],
    )
    def test_input_validation(self, inputs):
        x, y, z = 0.0, 0.0, 0.0

        with pytest.raises(ValueError):
            EmitterResponse(x, y, z, **inputs)


class TestConstantEmitters:
    @pytest.fixture
    def emitter_data(self):
        return {
            "x": [0.0, 1.0, -0.5],
            "y": [0.0, -0.5, -0.5],
            "z": [-0.5, -0.5, 0.1],
            "rate": 200,
            "wavelength": 700,
        }

    def test_ConstantEmitters_response(self, emitter_data):
        source = create_autospec(Source, spec_set=True)
        time = 0
        dt = 0.1
        emitters = ConstantEmitters(**emitter_data)

        response = emitters.response(time, dt, source)

        assert len(emitter_data["x"]) == len(response)
        # Check the number of emitted photons is correct for this time interval.
        assert all(r.photons == dt * emitter_data["rate"] for r in response)

    @pytest.mark.parametrize(
        "inputs",
        [
            {"rate": -100, "wavelength": 532.0},
            {"rate": 100, "wavelength": -532.0},
            {"rate": -100, "wavelength": -532.0},
        ],
    )
    def test_ConstantEmitters_bad_inputs(self, inputs):
        with pytest.raises(ValueError):
            ConstantEmitters(0.0, 0.0, 0.0, **inputs)


class TestFluorophore:
    @dataclass
    class DummySource(Source):
        irradiance_return_value: float

        def irradiance(self, *args) -> float:
            return self.irradiance_return_value

    def photon_rate(
        self,
        irradiance: float,
        cross_section: float,
        quantum_yield: float,
        fluorescence_lifetime: float,
    ):
        return (
            quantum_yield
            * cross_section
            * irradiance
            / (1 + irradiance * cross_section * quantum_yield * fluorescence_lifetime)
        )

    @pytest.mark.usefixtures("fluorophore")
    def test_fluorophore_response(self, fluorophore):
        # 0.6 time units spent in the fluorescence state 0
        events = [
            Event(time=0.2, from_state=0, to_state=1),
            Event(time=0.5, from_state=1, to_state=0),
            Event(time=0.9, from_state=0, to_state=2),
        ]
        fluorophore.state_machine.collect.return_value = events
        irradiance = 3.2e10  # photons / time / area
        expected_photons = round(
            0.6
            * self.photon_rate(
                irradiance,
                fluorophore.cross_section,
                fluorophore.quantum_yield,
                fluorophore.fluorescence_lifetime,
            )
        )
        source = self.DummySource(3.2e10)  # photons / time / area
        response = fluorophore.response(0, 1, source)

        assert expected_photons == response.photons

    @pytest.mark.usefixtures("fluorophore")
    def test_fluorophore_compute_photon_rate(self, fluorophore):
        irradiance = 3.2e10  # photons / time / area
        expected = self.photon_rate(
            irradiance,
            fluorophore.cross_section,
            fluorophore.quantum_yield,
            fluorophore.fluorescence_lifetime,
        )

        assert expected == fluorophore.compute_photon_rate(irradiance)

    @pytest.mark.usefixtures("fluorophore")
    def test_fluorophore_compute_photon_rate_0_irradiance(self, fluorophore):
        irradiance = 0

        assert 0 == fluorophore.compute_photon_rate(irradiance)

    @pytest.mark.usefixtures("fluorophore")
    def test_flourophore_compute_on_fraction(self, fluorophore):
        fluorophore.fluorescence_state = 0
        # 0.6 time units spent in the fluorescence state 0 (0.2 + 0.4)
        events = [
            Event(time=0.2, from_state=0, to_state=1),
            Event(time=0.5, from_state=1, to_state=0),
            Event(time=0.9, from_state=0, to_state=2),
        ]

        np.testing.assert_almost_equal(
            fluorophore.compute_on_fraction(0, 1, events), 0.6
        )

    @pytest.mark.usefixtures("fluorophore")
    def test_flourophore_compute_on_fraction_no_events_on_state(self, fluorophore):
        fluorophore.fluorescence_state = 0
        fluorophore.state_machine.current_state = 0
        events = []

        assert 1 == fluorophore.compute_on_fraction(0, 1, events)

    @pytest.mark.usefixtures("fluorophore")
    def test_flourophore_compute_on_fraction_no_events_off_state(self, fluorophore):
        fluorophore.fluorescence_state = 0
        fluorophore.state_machine.current_state = 1
        events = []

        assert 0 == fluorophore.compute_on_fraction(0, 1, events)

    @pytest.mark.parametrize(
        "inputs",
        [
            {
                "cross_section": -1e-6,
                "fluorescence_lifetime": 1e-6,
                "fluorescence_state": 0,
                "quantum_yield": 0.8,
                "wavelength": 7,
            },
            {
                "cross_section": 1e-6,
                "fluorescence_lifetime": -1e-6,
                "fluorescence_state": 0,
                "quantum_yield": 0.8,
                "wavelength": 7,
            },
            {
                "cross_section": 1e-6,
                "fluorescence_lifetime": 1e-6,
                "fluorescence_state": -1,
                "quantum_yield": 0.8,
                "wavelength": 7,
            },
            {
                "cross_section": 1e-6,
                "fluorescence_lifetime": 1e-6,
                "fluorescence_state": 0,
                "quantum_yield": -0.8,
                "wavelength": 7,
            },
            {
                "cross_section": 1e-6,
                "fluorescence_lifetime": 1e-6,
                "fluorescence_state": 0,
                "quantum_yield": 1.8,
                "wavelength": 7,
            },
            {
                "cross_section": 1e-6,
                "fluorescence_lifetime": 1e-6,
                "fluorescence_state": 0,
                "quantum_yield": 0.8,
                "wavelength": -7,
            },
        ],
    )
    def test_fluorophore_bad_inputs(self, inputs):
        state_machine = create_autospec(StateMachine)
        with pytest.raises(ValueError):
            Fluorophore(0, 0, 0, state_machine=state_machine, **inputs)
