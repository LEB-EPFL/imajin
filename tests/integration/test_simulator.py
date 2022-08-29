from leb.imajin.detectors import SimpleCMOSCamera
from leb.imajin.optics import Gaussian2D, SimpleMicroscope
from leb.imajin.samples import ConstantEmitters
from leb.imajin.simulators import Simulator
from leb.imajin.sources import UniformMono2D


def test_simple_simulation():
    # Take 100 images at 0.01 intervals per unit time on a 32 pixel x 32 pixel grid
    dt = 0.01
    x_lim = (0, 32)
    y_lim = (0, 32)
    num_measurements = 100

    # Create a uniform source of monochromatic illumination covering the computational space with
    # a maximum power of 100 mW
    source = UniformMono2D(0.1, 0.01, x_lim, y_lim)

    # Create an emitter at (16, 16, 0.05) emitting 10e5 photons per unit time
    sample = ConstantEmitters([16], [16], [0.05], 10e5, 0.7e-6)

    # Create a microscope with a Gaussan PSF 3 pixel in width
    psf = Gaussian2D(fwhm=3)
    optics = SimpleMicroscope(psf=psf)

    # Create a CMOS camera with a sensor size of 32 pixels x 32 pixels
    detector = SimpleCMOSCamera(num_pixels=(x_lim[1], y_lim[1]))

    # Prepare the simulation and run it
    simulator = Simulator(
        detector, optics, sample, source, dt, x_lim, y_lim, num_measurements
    )
    measurements = simulator.run()

    assert measurements.shape == (x_lim[1], y_lim[1], num_measurements)
    assert (measurements > 0).all()
