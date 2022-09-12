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


def test_three_state_flourophores_simulation():
    # Take 100 images at 0.01 intervals per unit time on a 32 pixel x 32 pixel grid starting at
    # time = 0
    time = 0.0
    dt = 1
    x_lim = (0, 32)
    y_lim = (0, 32)
    num_measurements = 100

    rng = np.random.default_rng()

    # Create a uniform source of monochromatic illumination covering the computational space with
    # a maximum power of 1e13 photons / sec
    source = UniformMono2D(1e13, 1e12, x_lim, y_lim)

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

    # Create a microscope with a Gaussan PSF 3 pixel in width
    psf = Gaussian2D(fwhm=3)
    optics = SimpleMicroscope(psf=psf)

    # Create a CMOS camera with a sensor size of 32 pixels x 32 pixels
    detector = SimpleCMOSCamera(num_pixels=(x_lim[1], y_lim[1]))

    # Prepare the simulation and run it
    simulator = Simulator(
        detector,
        optics,
        sample,
        source,
        time,
        dt,
        x_lim,
        y_lim,
        num_measurements,
        rng=rng,
    )
    measurements = simulator.run()

    assert measurements.shape == (num_measurements, x_lim[1], y_lim[1])
    assert (measurements > 0).all()
    np.testing.assert_approx_equal(time + dt * num_measurements, simulator.time)
