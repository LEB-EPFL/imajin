import numpy as np

from leb.imajin.detectors import SimpleCMOSCamera
from leb.imajin.optics import Gaussian2D, SimpleMicroscope
from leb.imajin.samples import ConstantEmitters
from leb.imajin.samples.factories import SampleType, sample_factory
from leb.imajin.simulators import Simulator
from leb.imajin.sources import UniformMono2D


def test_simple_simulation():
    # Take 100 images at 0.01 intervals per unit time on a 32 pixel x 32 pixel grid starting at
    # time = 0
    time = 0.0
    dt = 0.01
    x_lim = (0, 32)
    y_lim = (0, 32)
    num_measurements = 100

    # Create a uniform source of monochromatic illumination covering the computational space with
    # a maximum power of 10000 photons / sec
    source = UniformMono2D(1e4, 1e3, x_lim, y_lim)

    # Create an emitter at (16, 16, 0.05) emitting 10e5 photons per unit time
    sample = ConstantEmitters([16], [16], [0.05], 10e5, 0.7e-6)

    # Create a microscope with a Gaussan PSF 3 pixel in width
    psf = Gaussian2D(fwhm=3)
    optics = SimpleMicroscope(psf=psf)

    # Create a CMOS camera with a sensor size of 32 pixels x 32 pixels
    detector = SimpleCMOSCamera(num_pixels=(x_lim[1], y_lim[1]))

    # Prepare the simulation and run it
    simulator = Simulator(
        detector, optics, sample, source, time, dt, x_lim, y_lim, num_measurements
    )
    measurements = simulator.run()

    assert measurements.shape == (num_measurements, x_lim[1], y_lim[1])
    assert (measurements > 0).all()
    np.testing.assert_approx_equal(time + dt * num_measurements, simulator.time)

    # Reset the state of the simulation
    simulator.reset()
    np.testing.assert_approx_equal(0, simulator.time)
