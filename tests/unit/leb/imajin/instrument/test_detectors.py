import numpy as np
import pytest

from leb.imajin.instruments import CMOSCamera


@pytest.mark.parametrize(
    "inputs",
    [
        {"qe": -0.5, "sensitivity": 5.88, "dark_noise": 2.29, "baseline": 100},
        {"qe": 1.5, "sensitivity": 5.88, "dark_noise": 2.29, "baseline": 100},
        {"qe": 0.69, "sensitivity": -5.9, "dark_noise": 2.29, "baseline": 100},
        {"qe": 0.69, "sensitivity": 5.88, "dark_noise": -2.3, "baseline": 100},
        {"qe": 0.69, "sensitivity": 5.88, "dark_noise": 2.29, "baseline": -50},
    ],
)
def test_CMOSCamera_bad_intrinsic_parameter_values(inputs):
    with pytest.raises(ValueError):
        CMOSCamera(**inputs)


def test_CMOSCamera_snapshot():
    num_pixels = (128, 128)

    camera = CMOSCamera()
    photons = 1000 * np.ones(num_pixels)

    img = camera.snapshot(photons)

    assert img.shape == num_pixels
