from unittest.mock import create_autospec

import numpy as np
import numpy.typing as npt
import pytest

from leb.imajin.detectors import SimpleCMOSCamera
from leb.imajin.optics import Gaussian2D, SimpleMicroscope
from leb.imajin.samples.factories import SampleType, sample_factory
from leb.imajin.simulators import Processor, Simulator
from leb.imajin.sources import UniformMono2D


@pytest.fixture
def simulator():
    time = 0.0
    dt = 1
    x_lim = (0, 32)
    y_lim = (0, 32)
    num_measurements = 100

    source = UniformMono2D(1e4, 1e3, x_lim, y_lim)

    y, x = np.mgrid[5:30:5, 5:30:5]
    z = np.zeros(y.shape)
    positions = np.vstack([x.ravel(), y.ravel(), z.ravel()]).T
    sample = sample_factory(
        sample_type=SampleType.THREE_STATE_FLUOROPHORES,
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        k_off=1e-11,
        k_bleach=1e-14,
    )

    psf = Gaussian2D(fwhm=2.5)
    optics = SimpleMicroscope(psf=psf)

    detector = SimpleCMOSCamera(num_pixels=(x_lim[1], y_lim[1]))

    simulator = Simulator(
        detector, optics, sample, source, time, dt, x_lim, y_lim, num_measurements
    )

    return simulator


@pytest.fixture
def preprocessor():
    return create_autospec(Processor, spec_set=True)


@pytest.fixture
def post_processor():
    return create_autospec(Processor, spec_set=True)
