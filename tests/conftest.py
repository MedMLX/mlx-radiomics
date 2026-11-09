"""Shared test helpers: convention-set settings and the FFT availability marker."""

from dataclasses import replace
from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.shape import shape_statistics
from mlxradiomics.filters import FFT_AVAILABLE, FFT_DISABLED_REASON
from mlxradiomics.numerics.runtime import mx

requires_fft = pytest.mark.skipif(not FFT_AVAILABLE, reason=FFT_DISABLED_REASON)

# Named convention sets exercised by parametrized tests; "ibsi" is the Settings default.
# "pyradiomics" selects the non-default value of each convention-dependent Settings field.
type Conventions = Literal["pyradiomics", "ibsi"]


def settings_for(conventions: Conventions, **fields: object) -> Settings:
    """Settings with ``fields`` under one named convention set (IBSI = the defaults)."""
    if conventions == "ibsi":
        return replace(Settings(), **fields)
    # The non-default option for every convention-dependent public Settings field.
    return replace(Settings(
        covariance_divisor="n", ellipsoid_area="exact", kurtosis="raw",
        zero_variance_moments="nan", ngtdm_singular_cases="unadjusted",
        neighborhood_support="distances", bin_origin="zero-lattice", constant_roi_bin="middle",
        mask_extrapolation="nearest", spline_boundary="mirror",
    ), **fields)


def convention_shape_statistics(
    conventions: Conventions, mask: NDArray[np.bool_], spacing: tuple[float, ...],
    names: tuple[str, ...],
) -> dict[str, mx.array]:
    """Run ``shape_statistics`` with the shape conventions of one named convention set."""
    settings = settings_for(conventions)
    return shape_statistics(
        mask, spacing, 64, names, covariance_divisor=settings.covariance_divisor,
        ellipsoid_area=settings.ellipsoid_area,
    )
