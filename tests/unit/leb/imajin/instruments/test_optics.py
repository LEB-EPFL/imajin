import numpy as np
import pytest
from scipy import special  # type: ignore

from leb.imajin.instruments.optics.psfs import Gaussian2D


def gauss2d(x=0, y=0, x0=0, y0=0, sx=1, sy=1):
    return (
        1.0
        / (2.0 * np.pi * sx * sy)
        * np.exp(
            -((x - x0) ** 2.0 / (2.0 * sx**2.0) + (y - y0) ** 2.0 / (2.0 * sy**2.0))
        )
    )


def erf2d(x=0, y=0, x0=0, y0=0, sx=1, sy=1):
    """Computes the probability of the intersection of a 2D normal distribution and a square area.

    This function models the proportion of a PSF that intersects a square pixel with a center
    located at (x,y). The PSF is modeled as a 2D normal distribution centered at (x0, y0).

    """
    sqrt2 = np.sqrt(2)
    return (
        0.25
        * (
            special.erf((x - x0 + 0.5) / sx / sqrt2)
            - special.erf((x + x0 - 0.5) / sx / sqrt2)
        )
        * (
            special.erf((y - y0 + 0.5) / sy / sqrt2)
            - special.erf((y + y0 - 0.5) / sy / sqrt2)
        )
    )


@pytest.mark.parametrize("psf_centers", [{"x0": 0, "y0": 0}])
def test_Gaussian2D_bin(psf_centers):
    y, x = np.ogrid[-3:3:7j, -3:3:7j]
    fwhm = 1.5
    expected = erf2d(x=x, y=y, sx=fwhm / 2.3548, sy=fwhm / 2.3548, **psf_centers)
    psf = Gaussian2D(fwhm=fwhm)

    binned = psf.bin(x, y)

    np.testing.assert_array_almost_equal(expected, binned)


@pytest.mark.parametrize("psf_centers", [{"x0": 0, "y0": 0}])
def test_Gaussian2D_sample(psf_centers):
    y, x = np.ogrid[-3:3:7j, -3:3:7j]
    fwhm = 1.5
    expected = gauss2d(x=x, y=y, sx=fwhm / 2.3548, sy=fwhm / 2.3548, **psf_centers)
    psf = Gaussian2D(fwhm=fwhm)

    sampled = psf.sample(x, y)

    np.testing.assert_array_almost_equal(expected, sampled)


@pytest.mark.parametrize("fwhm", [0, -1])
def test_Gaussian2D_fwhm_must_be_positive(fwhm):
    with pytest.raises(ValueError):
        Gaussian2D(fwhm=fwhm)
