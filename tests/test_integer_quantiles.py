"""Source-pair percentiles retain their compensated residual on integer populations."""

import numpy as np

from mlxradiomics.features.integer_quantiles import integer_source_quantiles
from mlxradiomics.numerics.runtime import mx


def test_integer_quantiles_preserve_compensated_fraction_and_masked_populations() -> None:
    # Five equally frequent integers give percentiles [1.4, 2, 3, 4, 4.6].
    # The fractions 7/5 and 23/5 have nonzero source-pair residuals after rounding
    # their high part to float32: exactly 1/41943040 and 1/10485760, respectively.
    # The second row is constant and masked tail values lie outside the anchor range.
    values = mx.array([[1, 2, 3, 4, 5, 100, 100, 100], [-1] * 3 + [100] * 5], dtype=mx.float32)
    valid = mx.array([[True] * 5 + [False] * 3, [True] * 3 + [False] * 5], dtype=mx.bool_)
    count = mx.array([5, 3], dtype=mx.int32)
    high, low, minimum, maximum = integer_source_quantiles(
        values, mx.zeros(values.shape, dtype=mx.float32), valid, count,
        minimum=-1, maximum=5,
    )
    np.testing.assert_array_equal(
        np.asarray(high, dtype=np.float32),
        np.array([[1.4, 2, 3, 4, 4.6], [-1, -1, -1, -1, -1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        np.asarray(low, dtype=np.float32),
        np.array([[1 / 41943040, 0, 0, 0, 1 / 10485760], [0] * 5], dtype=np.float32),
    )
    np.testing.assert_array_equal(np.asarray(minimum, dtype=np.float32), [1, -1])
    np.testing.assert_array_equal(np.asarray(maximum, dtype=np.float32), [5, -1])
