from unittest.mock import create_autospec

import numpy as np
import pytest
from scipy import special  # type: ignore

from leb.imajin import EmitterResponse, SampleResponse
from leb.imajin.optics import Gaussian2D, SimpleMicroscope


def gauss2d(x=0, y=0, x0=0, y0=0, sx=1, sy=1):
    return (
        1.0
        / (2.0 * np.pi * sx * sy)
        * np.exp(-((x - x0) ** 2.0 / (2.0 * sx**2.0) + (y - y0) ** 2.0 / (2.0 * sy**2.0)))
    )


def erf2d(x=0, y=0, x0=0, y0=0, sx=1, sy=1):
    """Computes the probability of the intersection of a 2D normal distribution and a square area.

    This function models the proportion of a PSF that intersects a square pixel with an upper left
    corner located at (x,y). The PSF is modeled as a 2D normal distribution centered at (x0, y0).

    """
    sqrt2 = np.sqrt(2)
    return (
        0.25
        * (special.erf((x - x0 + 1) / sx / sqrt2) - special.erf((x - x0) / sx / sqrt2))
        * (special.erf((y - y0 + 1) / sy / sqrt2) - special.erf((y - y0) / sy / sqrt2))
    )


class TestGaussian2D:
    @pytest.mark.parametrize("psf_centers", [{"x0": np.float64(0), "y0": np.float64(0)}])
    def test_Gaussian2D_bin(self, psf_centers):
        y, x = np.ogrid[-3:3:7j, -3:3:7j]
        fwhm = 1.5
        expected = erf2d(x=x, y=y, sx=fwhm / 2.3548, sy=fwhm / 2.3548, **psf_centers)
        psf = Gaussian2D(fwhm=fwhm)

        binned = psf.bin(x, y, **psf_centers)

        np.testing.assert_array_almost_equal(expected, binned)

    @pytest.mark.parametrize("psf_centers", [{"x0": np.float64(0), "y0": np.float64(0)}])
    def test_Gaussian2D_sample(self, psf_centers):
        y, x = np.ogrid[-3:3:7j, -3:3:7j]
        fwhm = 1.5
        expected = gauss2d(x=x, y=y, sx=fwhm / 2.3548, sy=fwhm / 2.3548, **psf_centers)
        psf = Gaussian2D(fwhm=fwhm)

        sampled = psf.sample(x, y, **psf_centers)

        np.testing.assert_array_almost_equal(expected, sampled)

    @pytest.mark.parametrize("fwhm", [0, -1])
    def test_Gaussian2D_fwhm_must_be_positive(self, fwhm):
        with pytest.raises(ValueError):
            Gaussian2D(fwhm=fwhm)


class TestSimpleMicroscope:
    @pytest.fixture
    def microscope(self):
        psf = Gaussian2D(fwhm=2)
        return SimpleMicroscope(psf=psf)

    @pytest.fixture
    def sample_response(self):
        return [
            EmitterResponse(4.0, 4.0, 0.0, 100, 0.7e-6),
            EmitterResponse(5.0, 7.0, 0.0, 1000, 0.7e-6),
        ]

    def test_response(self, sample_response):
        x_lim = (0, 32)  # Units are pixels
        y_lim = (0, 32)
        psf = Gaussian2D(fwhm=3)
        microscope = SimpleMicroscope(psf)

        photons = microscope.response(x_lim, y_lim, sample_response)

        assert photons.shape == (y_lim[1], x_lim[1])

        # The sum of all photons should be equal to the photons emitted by all fluorophores, with a
        # small error due to PSF clipping at the edges. Expected 1100 without clipping.
        np.testing.assert_approx_equal(
            np.sum(photons), sum(r.photons for r in sample_response), significant=3
        )

    @pytest.mark.parametrize(
        "spatial_limits",
        [
            {"x_lim": (5, 4), "y_lim": (2, 3)},
            {"x_lim": (4, 5), "y_lim": (3, 2)},
            {"x_lim": (5, 4), "y_lim": (3, 2)},
        ],
    )
    def test_bad_spatial_limits(self, spatial_limits):
        psf = Gaussian2D(fwhm=3)
        sample_response = create_autospec(SampleResponse, spec_set=True)
        microscope = SimpleMicroscope(psf)

        with pytest.raises(ValueError):
            microscope.response(spatial_limits["x_lim"], spatial_limits["y_lim"], sample_response)

    def test_psf_clipping(self, microscope):
        """Emitters near the border of the computational grid should have their responses scaled."""
        x_lim = (0, 16)
        y_lim = (0, 16)

        # An emitter at (0, 0) with a rotationally-symmetric PSF will have 3/4 of its number of photons
        # lost due to clipping of its image because the camera's upper left corner is at (0, 0).
        photons = 100
        expected_num_photons = photons / 4
        sample_response = [EmitterResponse(x=0, y=0, z=0, photons=photons, wavelength=700)]

        optics_response = microscope.response(x_lim, y_lim, sample_response)

        assert expected_num_photons == optics_response.sum()
