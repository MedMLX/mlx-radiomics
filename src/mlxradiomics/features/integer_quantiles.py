"""Bounded integer source quantiles with the source-pair interpolation contract.

A validated source descriptor admits only finite integral values in [-2**24, 2**24],
with no sampled negative zero and a lattice of at most 65536 levels. Every admitted
integer is exactly representable in float32: the source pair therefore has
that integer as its high part and a zero residual. Host bounds only anchor the integer
histogram; ranks, selected endpoint values and extrema are determined on the GPU.
"""

from mlxradiomics.features.intensity_histogram import histogram_order_values, integer_histogram
from mlxradiomics.numerics.precision import source_quantile_endpoints
from mlxradiomics.numerics.runtime import mx


def integer_source_quantiles(
    values: mx.array,
    residual: mx.array,
    valid: mx.array,
    count: mx.array,
    *,
    minimum: int,
    maximum: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Return source-precision quantile pairs and occupied extrema for integral sets.

    ``values`` and ``residual`` are the original float32 source pair, and ``valid``
    identifies sampled columns. The caller proves the admission contract above; its
    ``count`` is the exact int32 population of each row. Invalid source parts
    are never counted. ``maximum - minimum + 1`` must not exceed either the source row
    width or 65536, bounding count storage by the source rather than a sparse range.

    All ten adjacent percentile endpoints are selected by exact integer rank search.
    Their source pairs pass through the compensated quantile interpolation
    kernel, changing only endpoint addressing. In particular, this does not substitute
    the intensity-histogram family's integer-plus-fraction interpolation arithmetic.

    Returns ``(quantiles_high, quantiles_low, minimum, maximum)``. Quantile arrays have
    shape ``(batch, 5)`` (10/25/50/75/90); extrema have shape ``(batch,)``. Extrema are
    derived from occupied GPU counts, independently of the caller's anchor bounds.
    """
    levels = maximum - minimum + 1
    if not (-2**24 <= minimum <= maximum <= 2**24):
        raise ValueError("integer source quantiles require bounds within [-2**24, 2**24]")
    if levels > min(65536, values.shape[1]):
        raise ValueError("integer source quantile lattice exceeds its bounded source width")
    # Every admitted source integer and this bounded difference fit int32 exactly.
    # Invalid values can be arbitrary; the histogram clamps their zero-weight indexes.
    bins = values.astype(mx.int32) - minimum + 1
    counts = integer_histogram(bins, valid, levels)
    count_wide = count.astype(mx.int64)
    n = mx.maximum(count_wide - 1, 0)[:, None]
    numerator = mx.array([1, 1, 1, 3, 9], dtype=mx.int64)[None, :]
    denominator = mx.array([10, 4, 2, 4, 10], dtype=mx.int64)[None, :]
    lower = (n // denominator) * numerator + ((n % denominator) * numerator) // denominator
    upper = mx.minimum(lower + 1, n)
    ranks = mx.reshape(mx.stack([lower, upper], axis=2), (values.shape[0], 10))
    endpoints = histogram_order_values(counts, count_wide, ranks) + minimum - 1
    endpoint_high = endpoints.astype(mx.float32)
    # The admitted integer range is exactly float32-representable. Keeping the explicit
    # zero residual yields precisely the same source pairs as prepare_source_pair.
    endpoint_low = mx.zeros(endpoint_high.shape, dtype=residual.dtype)
    quantiles = source_quantile_endpoints(endpoint_high, endpoint_low, count)
    occupied = (counts > 0).astype(mx.int32)
    minimum_level = mx.argmax(occupied, axis=1).astype(mx.int32)
    maximum_level = levels - 1 - mx.argmax(mx.flip(occupied, axis=1), axis=1).astype(mx.int32)
    return (
        quantiles[0], quantiles[1],
        (minimum_level + minimum).astype(mx.float32),
        (maximum_level + minimum).astype(mx.float32),
    )
