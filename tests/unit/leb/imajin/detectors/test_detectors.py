from unittest.mock import create_autospec

import numpy as np
import pytest

from leb.imajin.detectors import BitDepth, SimpleCMOSCamera


@pytest.fixture
def rs_stub():
    """Returns a stub of a random number generator for deterministic testing."""
    return create_autospec(np.random.Generator, spec_set=True)


def test_SimpleCMOSCamera_response_correct_value(rs_stub):
    bit_depth = BitDepth.TWELVE
    sensitivity = 2
    photons_avg = 100
    camera = SimpleCMOSCamera(bit_depth=bit_depth, sensitivity=sensitivity)
    photons = photons_avg * np.ones(camera.num_pixels)

    rs_stub.poisson.return_value = (photons_avg + 10) * np.ones(camera.num_pixels)
    rs_stub.normal.return_value = 10

    img = camera.response(photons=photons, rng=rs_stub)

    # Pre-calculated result is 340 ADUs in each pixel
    assert np.all(340 == img)
    assert np.uint16 == img.dtype


def test_SimpleCMOSCamera_response_benchmark(benchmark):
    photons_avg = 100
    camera = SimpleCMOSCamera()
    photons = photons_avg * np.ones(camera.num_pixels)

    benchmark(SimpleCMOSCamera.response, camera, photons=photons)


def test_SimpleCMOSCamera_response_saturation(rs_stub):
    photons_avg = 1e10
    bit_depth = BitDepth.EIGHT
    camera = SimpleCMOSCamera(bit_depth=bit_depth)
    photons = photons_avg * np.ones(camera.num_pixels)

    rs_stub.poisson.return_value = (photons_avg + 0.1 * photons_avg) * np.ones(camera.num_pixels)
    rs_stub.normal.return_value = 10 * np.ones(camera.num_pixels)

    img = camera.response(photons=photons, rng=rs_stub)

    # All pixels saturated at the maximum value for an 8-bit sensor
    assert np.all(255 == img)


def test_SimpleCMOSCamera_response_output_size_is_correct_with_signal():
    num_pixels = (128, 128)

    camera = SimpleCMOSCamera(num_pixels=num_pixels)
    photons = 1000 * np.ones(num_pixels)

    img = camera.response(photons)

    assert img.shape == num_pixels


def test_SimpleCMOSCamera_response_output_size_is_correct_without_signal():
    num_pixels = (128, 128)

    camera = SimpleCMOSCamera(num_pixels=num_pixels)
    photons = None

    img = camera.response(photons=photons)

    assert img.shape == num_pixels


def test_SimpleCMOSCamera_response_negative_photons():
    photons = np.array([[-100, 150], [100, -100]])
    camera = SimpleCMOSCamera()

    with pytest.raises(ValueError):
        camera.response(photons)


def test_SimpleCMOSCamera_response_photons_wrong_shape():
    photons = np.ones((32, 32))
    num_pixels = (64, 64)
    camera = SimpleCMOSCamera(num_pixels=num_pixels)

    with pytest.raises(ValueError):
        camera.response(photons)


@pytest.mark.parametrize(
    "inputs",
    [
        {
            "baseline": 100,
            "dark_noise": 2.29,
            "num_pixels": (64, 64),
            "quantum_efficiency": -0.5,
            "sensitivity": 5.88,
        },
        {
            "baseline": 100,
            "dark_noise": 2.29,
            "num_pixels": (64, 64),
            "quantum_efficiency": 1.5,
            "sensitivity": 5.88,
        },
        {
            "baseline": 100,
            "dark_noise": 2.29,
            "num_pixels": (64, 64),
            "quantum_efficiency": 0.69,
            "sensitivity": -5.9,
        },
        {
            "baseline": 100,
            "dark_noise": -2.3,
            "num_pixels": (64, 64),
            "quantum_efficiency": 0.69,
            "sensitivity": 5.88,
        },
        {
            "baseline": -50,
            "dark_noise": 2.29,
            "num_pixels": (64, 64),
            "quantum_efficiency": 0.69,
            "sensitivity": 5.88,
        },
        {
            "baseline": -50,
            "dark_noise": 2.29,
            "num_pixels": (-64, 64),
            "quantum_efficiency": 0.69,
            "sensitivity": 5.88,
        },
        {
            "baseline": -50,
            "dark_noise": 2.29,
            "num_pixels": (64, -64),
            "quantum_efficiency": 0.69,
            "sensitivity": 5.88,
        },
        {
            "baseline": -50,
            "dark_noise": 2.29,
            "num_pixels": (-64, -64),
            "quantum_efficiency": 0.69,
            "sensitivity": 5.88,
        },
    ],
)
def test_CMOSCamera_bad_parameter_values(inputs):
    with pytest.raises(ValueError):
        SimpleCMOSCamera(**inputs)
