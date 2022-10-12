"""Benchmark Test 0: Three state fluorophores"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pytest

from leb.imajin import PSF, Source
from leb.imajin.detectors import SimpleCMOSCamera
from leb.imajin.optics import Gaussian2D, SimpleMicroscope
from leb.imajin.samples.factories import SampleType, sample_factory
from leb.imajin.simulators import Simulator
from leb.imajin.sources import UniformMono2D


@pytest.fixture(scope="session")
def reset(pytestconfig):
    return pytestconfig.getoption("reset")


@dataclass(frozen=True)
class Parameters:
    time: float = 0.0
    dt: float = 1.0
    x_lim: Tuple[int, int] = (0, 32)
    y_lim: Tuple[int, int] = (0, 32)
    num_measurements: int = 10


def new_simulator() -> Simulator:
    # Take 100 images at 0.01 intervals per unit time on a 32 pixel x 32 pixel grid starting at
    # time = 0

    rng = np.random.default_rng()

    # Create a uniform source of monochromatic illumination covering the computational space with
    # a maximum power of 1e15 photons / sec
    source: Source = UniformMono2D(1e15, 1e14, Parameters.x_lim, Parameters.y_lim)

    # Create a 5x5 grid of fluorophores that can blink on and off and bleach
    y, x = np.mgrid[5:30:5, 5:30:5]
    z = np.zeros(y.shape)
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    sample = sample_factory(
        sample_type=SampleType.THREE_STATE_FLUOROPHORES,
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        k_off=1e-11,
        k_bleach=1e-15,
        rng=rng,
    )

    # Create a microscope with a Gaussan PSF 2 pixels in width
    psf: PSF = Gaussian2D(fwhm=2)
    optics = SimpleMicroscope(psf=psf)

    # Create a CMOS camera with a sensor size of 32 pixels x 32 pixels
    detector = SimpleCMOSCamera(num_pixels=(Parameters.x_lim[1], Parameters.y_lim[1]))

    # Prepare the simulation and run it
    return Simulator(
        detector,
        optics,
        sample,
        source,
        Parameters.time,
        Parameters.dt,
        Parameters.x_lim,
        Parameters.y_lim,
        Parameters.num_measurements,
        rng=rng,
    )


def test_three_state_flourophores_simulation(benchmark, reset):
    simulator = new_simulator()
    measurements = benchmark(Simulator.run, simulator, reset=reset)

    assert measurements.shape == (
        Parameters.num_measurements,
        Parameters.x_lim[1],
        Parameters.y_lim[1],
    )
    assert (measurements > 0).all()
    if reset is False:
        np.testing.assert_approx_equal(
            Parameters.time + Parameters.dt * Parameters.num_measurements, simulator.time
        )


if __name__ == "__main__":
    from pathlib import Path

    from viztracer import VizTracer  # type: ignore

    filename = Path(__file__).stem + "_result.json"

    simulator = new_simulator()

    with VizTracer(output_file=filename) as tracer:
        simulator.run()
