"""Filter choices and scalar parameter validation."""

from enum import StrEnum
from math import (
    isfinite,
    sqrt,
)
from typing import cast


class FilterPadding(StrEnum):
    """IBSI 2 boundary-extension methods (manual section 2.2).

    ``MIRROR`` repeats the boundary voxel (NumPy/MLX ``symmetric``), which is the
    boundary-inclusive mirror convention required by the manual.
    """

    CONSTANT = "constant"
    NEAREST = "nearest"
    PERIODIC = "periodic"
    MIRROR = "mirror"


class FourierGrid(StrEnum):
    """Sampling convention for full-image Fourier-domain filters.

    ``DFT`` uses ``2*pi*fftfreq(N)``, the manual's ``2*pi/N`` step, and has an
    exact zero-frequency sample. ``CENTERED`` uses
    ``(k - (N - 1) / 2) / ((N - 1) / 2)`` for ``k = 0, ..., N - 1``, which is
    endpoint-inclusive and has half-bin offsets for even lengths. The centered
    grid is the convention behind the published IBSI 2 consensus maps.
    """

    DFT = "dft"
    CENTERED = "centered"


class LogKernel(StrEnum):
    """Laplacian-of-Gaussian construction for ``ImageType.LOG``.

    ``RECURSIVE`` is ITK's recursive Gaussian approximation used by PyRadiomics,
    with its own constant-edge handling. ``SAMPLED`` is the IBSI 2 spatial kernel
    (manual Eq. 4.9) with an explicit cutoff and boundary condition, the
    construction behind the consensus maps ``2_a`` to ``2_c``.
    """

    RECURSIVE = "recursive"
    SAMPLED = "sampled"


class RieszBase(StrEnum):
    """Base filter to which the Riesz transform is applied.

    ``LOG`` is the IBSI 2 spatial Laplacian-of-Gaussian kernel; ``SIMONCELLI``
    is the radial Simoncelli B-map, which is periodic by construction.
    """

    LOG = "log"
    SIMONCELLI = "simoncelli"


class GaborResponse(StrEnum):
    """Which component of the complex Gabor response map to return.

    Values follow the IBSI 2 response-map IDs: modulus (``0J1M``), real
    (``F7DI``) and imaginary (``LI3Y``).
    """

    MODULUS = "modulus"
    REAL = "real"
    IMAGINARY = "imaginary"


class OrientationPooling(StrEnum):
    """Pooling across rotated versions of a directionally sensitive filter."""

    NONE = "none"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"


def validate_filter_constant(value: object, name: str) -> float:
    """Return a finite float filter-extension constant, or raise ``ValueError``."""
    if type(value) not in (int, float) or not isfinite(cast(float, value)):
        raise ValueError(f"{name} must be a finite number")
    return float(cast(float, value))


class LBPMethod(StrEnum):
    """LBP coding methods; values are PyRadiomics' ``lbp2DMethod`` names.

    ``DEFAULT`` is the raw bit code, ``ROTATION_INVARIANT`` the minimum over bit rotations,
    ``UNIFORM`` the count of set bits (or ``samples + 1`` for non-uniform patterns),
    ``NONROTATION_UNIFORM`` a uniform code that keeps rotation information, and
    ``VARIANCE`` the variance of the sample-minus-center differences (NaN where it is 0).
    """

    DEFAULT = "default"
    ROTATION_INVARIANT = "ror"
    UNIFORM = "uniform"
    NONROTATION_UNIFORM = "nri_uniform"
    VARIANCE = "var"


def validate_laws_kernel(codes: str) -> None:
    """Validate a concatenated Laws kernel name such as ``"E5L5S5"``.

    Args:
        codes: Two or three two-character kernel names in IBSI axis order
            ``(k1, k2, k3)``. ``k1`` is the last array axis.

    Raises:
        ValueError: If a name is unknown or the axis count is not two or three.
    """
    if len(codes) not in (4, 6) or any(
        codes[index : index + 2] not in LAWS_TAPS for index in range(0, len(codes), 2)
    ):
        raise ValueError(
            "laws_kernel must be two or three names from "
            f"{sorted(LAWS_TAPS)} concatenated in k1, k2, k3 order"
        )


MAX_STEERING_ORDER = 16


LAWS_TAPS: dict[str, tuple[float, ...]] = {
    "L3": (1 / sqrt(6), 2 / sqrt(6), 1 / sqrt(6)),
    "L5": (1 / sqrt(70), 4 / sqrt(70), 6 / sqrt(70), 4 / sqrt(70), 1 / sqrt(70)),
    "E3": (-1 / sqrt(2), 0.0, 1 / sqrt(2)),
    "E5": (-1 / sqrt(10), -2 / sqrt(10), 0.0, 2 / sqrt(10), 1 / sqrt(10)),
    "S3": (-1 / sqrt(6), 2 / sqrt(6), -1 / sqrt(6)),
    "S5": (-1 / sqrt(6), 0.0, 2 / sqrt(6), 0.0, -1 / sqrt(6)),
    "W5": (-1 / sqrt(10), 2 / sqrt(10), 0.0, -2 / sqrt(10), 1 / sqrt(10)),
    "R5": (1 / sqrt(70), -4 / sqrt(70), 6 / sqrt(70), -4 / sqrt(70), 1 / sqrt(70)),
}
