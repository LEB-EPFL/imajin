from typing import Iterable
from unittest.mock import create_autospec

import pytest

from leb.imajin.instruments import Source
from leb.imajin.samples import *


class TestEmitterToSource:
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
            EmitterToSource(x, y, z, **inputs)


class TestEmittersToSource:
    @pytest.fixture
    def emitters(self):
        return [
            EmitterToSource(0.0, 0.0, 0.0, 100, 700),
            EmitterToSource(1.0, -0.5, 0.2, 250, 690),
            EmitterToSource(-0.5, -0.5, 0.1, 110, 700),
        ]

    def test_EmittersToSource_is_iterable(self, emitters):
        response = EmittersToSource(emitters)

        assert isinstance(iter(response), Iterable)

    def test_EmittersToSource_length(self, emitters):
        response = EmittersToSource(emitters)

        assert len(response.emitters) == len(response)


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
        dt = 0.1
        emitters = ConstantEmitters(**emitter_data)

        response = emitters.response(source, dt)

        assert isinstance(response, EmittersToSource)
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
