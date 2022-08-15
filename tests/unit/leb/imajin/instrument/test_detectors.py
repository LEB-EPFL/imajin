import numpy as np
import pytest

from leb.imajin.instruments import SimpleCMOSCamera


@pytest.mark.parametrize(
    "inputs",
    [
        {"baseline": 100, "dark_noise": 2.29, "qe": -0.5, "sensitivity": 5.88},
        {"baseline": 100, "dark_noise": 2.29, "qe": 1.5, "sensitivity": 5.88},
        {"baseline": 100, "dark_noise": 2.29, "qe": 0.69, "sensitivity": -5.9},
        {"baseline": 100, "dark_noise": -2.3, "qe": 0.69, "sensitivity": 5.88},
        {"baseline": -50, "dark_noise": 2.29, "qe": 0.69, "sensitivity": 5.88},
    ],
)
def test_CMOSCamera_bad_intrinsic_parameter_values(inputs):
    with pytest.raises(ValueError):
        SimpleCMOSCamera(**inputs)


def test_CMOSCamera_snapshot():
    num_pixels = (128, 128)

    camera = SimpleCMOSCamera()
    photons = 1000 * np.ones(num_pixels)

    img = camera.get_image(photons)

    assert img.shape == num_pixels
