"""Host representations and admission proofs for ROI and neighborhood first order.

This module classifies already validated float64 samples; it never computes a
feature. Invalid grid positions remain present and hold zero so the GPU reduction
width, sample order and reduction tree match the padded ROI layout.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.numerics.precision import prepare_source_pair

_SIGN = np.uint64(1 << 63)
_MAGNITUDE = np.uint64((1 << 63) - 1)
_NORMAL_BITS = np.array(2.0**-126, dtype=np.float64).view(np.uint64)


@dataclass(frozen=True)
class FirstOrderAdmission:
    """Conservative source proofs for the GPU first-order feature graph.

    ``exact_pair`` includes Metal's subnormal flushing. ``source_normal`` proves
    no sampled nonzero source is below the normal float32 range. Integer bounds
    qualify algorithms only; reported extrema and all feature arithmetic stay GPU.
    ``bounded_integer_mean`` bounds source integer mass by 2**40; ROI admission
    conservatively uses padded width times maximum magnitude. ``integer_quantiles``
    additionally bounds histogram storage and refuses
    negative zero, whose source-order tie behavior is handled by the general ordering path.
    """

    source_normal: bool
    zero_residual: bool
    exact_pair: bool
    bounded_integer_mean: bool
    integer_quantiles: bool
    integer_minimum: int | None
    integer_maximum: int | None

    def neighborhood_admission(self) -> "FirstOrderAdmission":
        """Retain pointwise/mass proofs for unique in-image neighborhood subsets.

        Any subset retains normal sources, zero residuals and exact pairs. Its
        integer absolute mass is bounded by the complete source's admitted mass.
        Histogram/rank admission depends on each row's own layout and support, so
        neighborhoods always use the general ordering path instead.
        """
        return FirstOrderAdmission(
            source_normal=self.source_normal,
            zero_residual=self.zero_residual,
            exact_pair=self.exact_pair,
            bounded_integer_mean=self.bounded_integer_mean,
            integer_quantiles=False,
            integer_minimum=None,
            integer_maximum=None,
        )


@dataclass(frozen=True)
class FirstOrderSource(FirstOrderAdmission):
    """One padded source row and its admission, without compacting reduction order."""

    high: NDArray[np.float32]
    low: NDArray[np.float32]
    representation_error: NDArray[np.float32]
    source_bits: NDArray[np.uint64]
    row_width: int
    count: int


def prepare_first_order_source(
    image: NDArray[np.float64],
    valid: NDArray[np.bool_],
    *,
    sampled: NDArray[np.float64] | None = None,
    integer_bounds: tuple[int, int] | None = None,
) -> FirstOrderSource:
    """Prepare sampled source parts once while preserving the padded grid layout.

    ``sampled`` may reuse the exact, finite samples returned by preprocessing. Its
    NumPy boolean-mask order must match ``image[valid]``; removing an enclosing
    empty grid shell preserves that order. No source conversion or finite check is
    repeated here. ``integer_bounds`` may carry exact bounded-integral admission
    from preprocessing. Non-finite background pixels are never represented or classified.
    """
    samples = image[valid] if sampled is None else sampled
    bits = samples.view(np.uint64)
    row_width, count = int(image.size), int(samples.size)
    if integer_bounds is None:
        high, low, error = prepare_source_pair(samples)
        zero_residual = not bool(np.any(low))
        exact_pair = not bool(np.any(error))
        minimum, maximum = float(samples.min()), float(samples.max())
        finite = isfinite(minimum) and isfinite(maximum)
        integral = (
            finite and exact_pair and zero_residual and max(abs(minimum), abs(maximum)) <= 2**24
            and bool(np.all(high == high.astype(np.int32)))
        )
    else:
        # Binning has already proved every sample integral with |source| <=2**24.
        # Its float32 high part is exact and normal, and both omitted parts are zero.
        high = samples.astype(np.float32)
        low = np.zeros_like(high, dtype=np.float32)
        error = low
        minimum, maximum = integer_bounds
        finite = integral = zero_residual = exact_pair = True
    if exact_pair:
        # An exactly represented Metal-visible pair cannot hold a source below
        # 2**-126: both of its parts would flush, yielding a nonzero error bound.
        source_normal = True
    else:
        magnitude_bits = bits & _MAGNITUDE
        source_normal = finite and not bool(np.any(
            (magnitude_bits > 0) & (magnitude_bits < _NORMAL_BITS),
        ))
    integer_minimum: int | None = None
    integer_maximum: int | None = None
    bounded_integer_mean = False
    integer_quantiles = False
    if integral:
        integer_minimum, integer_maximum = int(minimum), int(maximum)
        bounded_integer_mean = row_width * max(abs(integer_minimum), abs(integer_maximum)) <= 2**40
        levels = integer_maximum - integer_minimum + 1
        integer_quantiles = (
            levels <= min(65536, row_width) and not bool(np.any(bits == _SIGN))
        )

    def padded_float(part: NDArray[np.float32]) -> NDArray[np.float32]:
        if count == row_width:
            return part.reshape(1, -1)
        padded = np.zeros(image.shape, dtype=np.float32)
        padded[valid] = part
        return padded.reshape(1, -1)

    if count == row_width:
        padded_bits = bits.reshape(1, -1)
    else:
        padded_bits = np.zeros(image.shape, dtype=np.uint64)
        padded_bits[valid] = bits
        padded_bits = padded_bits.reshape(1, -1)
    return FirstOrderSource(
        source_normal=source_normal,
        zero_residual=zero_residual,
        exact_pair=exact_pair,
        bounded_integer_mean=bounded_integer_mean,
        integer_quantiles=integer_quantiles,
        integer_minimum=integer_minimum,
        integer_maximum=integer_maximum,
        high=padded_float(high),
        low=padded_float(low),
        representation_error=padded_float(error),
        source_bits=padded_bits,
        row_width=row_width,
        count=count,
    )
