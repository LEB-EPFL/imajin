import numpy as np
import pytest

from leb.imajin.sources import UniformMono2D


def test_UniformMono2D_e_field():
    power = 0.1
    x_lim = -0.001, 0.001
    y_lim = -0.001, 0.001
    expected_e_field = np.sqrt(power / (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0]))
    light_source = UniformMono2D(power_max=50, power=power, x_lim=x_lim, y_lim=y_lim)

    light_source.e_field(0, 0) == expected_e_field


def test_UniformMono2D_e_field_in_dielectric():
    power = 0.1
    impedance = 376 / 1.33
    x_lim = -0.001, 0.001
    y_lim = -0.001, 0.001
    expected_e_field = np.sqrt(
        power * impedance / (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0])
    )
    light_source = UniformMono2D(power_max=50, power=power, x_lim=x_lim, y_lim=y_lim)

    light_source.e_field(0, 0, impedance=impedance) == expected_e_field


def test_UniformMono2D_irradiance():
    power = 0.1
    x_lim = -0.001, 0.001
    y_lim = -0.001, 0.001
    expected_irradiance = power / (x_lim[1] - x_lim[0]) / (y_lim[1] - y_lim[0])
    light_source = UniformMono2D(power_max=50, power=power, x_lim=x_lim, y_lim=y_lim)

    light_source.irradiance(0, 0) == expected_irradiance


@pytest.mark.parametrize("positions", [(-0.0005, -0.0005), (0, 0), (-0.001, 0.001)])
def test_UniformMono2D_in_bounds(positions):
    light_source = UniformMono2D(
        power_max=50, power=0.1, x_lim=(-0.001, 0.001), y_lim=(-0.001, 0.001)
    )

    assert np.abs(light_source.e_field(x=positions[0], y=positions[1])) > 0
    assert light_source.irradiance(x=positions[0], y=positions[1]) > 0


@pytest.mark.parametrize("positions", [(-0.0005, 0.002), (0.002, -0.0005), (10, -100)])
def test_UniformMono2D_out_of_bounds(positions):
    light_source = UniformMono2D(
        power_max=50, power=0.1, x_lim=(-0.001, 0.001), y_lim=(-0.001, 0.001)
    )

    assert light_source.e_field(x=positions[0], y=positions[1]) == 0
    assert light_source.irradiance(x=positions[0], y=positions[1]) == 0


def test_UniformMono2D_power_cannot_exceed_maximum():
    power_max = 50
    light_source = UniformMono2D(
        power_max=power_max, power=0, x_lim=(-1, 1), y_lim=(-1, 1)
    )

    with pytest.raises(ValueError):
        light_source.power = power_max + 1


def test_UniformMono2D_cannot_set_power_max():
    light_source = UniformMono2D(power_max=50, power=0, x_lim=(-1, 1), y_lim=(-1, 1))

    with pytest.raises(AttributeError):
        light_source.POWER_MAX = 100


@pytest.mark.parametrize(
    "bad_inputs",
    [
        {"power_max": 200, "power": -100, "x_lim": (-1, 1), "y_lim": (-1, 1)},
        {"power_max": 200, "power": 100, "x_lim": (1, -1), "y_lim": (-1, 1)},
        {"power_max": 200, "power": 100, "x_lim": (-1, 1), "y_lim": (1, -1)},
        {"power_max": 50, "power": 100, "x_lim": (-1, 1), "y_lim": (-1, 1)},
    ],
)
def test_UniformMono2D_bad_inputs(bad_inputs):
    with pytest.raises(ValueError):
        UniformMono2D(**bad_inputs)
