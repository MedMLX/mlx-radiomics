"""FFT-based filters refuse every request on an MLX runtime without the FFT fix.

MLX releases before 0.32.5 can overrun a batched FFT buffer and corrupt adjacent
arrays, so Simoncelli and all Riesz variants are disabled there. These tests check
which MLX versions enable the filters and exercise the refusal at the planning,
generator and public-extraction boundaries without running an FFT.
"""

import dataclasses
import re
from collections.abc import Callable

import numpy as np
import pytest

from mlxradiomics.config.filtering import FilterPadding, FourierGrid, RieszBase
from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import (
    ImageType,
    RieszSettings,
    SimoncelliSettings,
    TransformSettings,
)
from mlxradiomics.filters import (
    FFT_AVAILABLE,
    FFT_DISABLED_REASON,
    riesz_log_responses,
    riesz_simoncelli_responses,
    simoncelli_response,
)
from mlxradiomics.filters.spectral import fft_available_on
from mlxradiomics.imaging.transforms import feature_image_names, feature_images
from mlxradiomics.numerics.runtime import mx
from mlxradiomics.workflows.extraction import extract

_DISABLED = re.escape(FFT_DISABLED_REASON)
_REFUSING = pytest.mark.skipif(FFT_AVAILABLE, reason="MLX 0.32.5 or later is installed")


@pytest.mark.parametrize(
    ("version", "available"),
    [("0.32.4", False), ("0.31.2", False), ("0.32.5.dev1", False),
     ("0.32.5", True), ("0.33.0", True), ("1.0.0", True)],
)
def test_fft_filters_need_a_fixed_mlx_release(version: str, available: bool) -> None:
    # Development builds stay disabled: some predate the fix.
    assert fft_available_on(version) is available

_FFT_SETTINGS: tuple[tuple[str, TransformSettings], ...] = (
    (
        "simoncelli",
        TransformSettings(
            image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(levels=(1,))
        ),
    ),
    (
        "riesz-log",
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(orders=((1, 0, 0),), log_sigma=(1.0,)),
        ),
    ),
    (
        "riesz-simoncelli",
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(
                base=RieszBase.SIMONCELLI, orders=((1, 0, 0),), padding=FilterPadding.PERIODIC
            ),
        ),
    ),
    (
        "riesz-log-steered",
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(
                orders=((1, 0, 0),), log_sigma=(1.0,), steering=True, tensor_sigma=1.0
            ),
        ),
    ),
    (
        "riesz-simoncelli-steered",
        TransformSettings(
            image_types=(ImageType.RIESZ,),
            riesz=RieszSettings(
                base=RieszBase.SIMONCELLI,
                orders=((1, 0, 0),),
                padding=FilterPadding.PERIODIC,
                steering=True,
                tensor_sigma=1.0,
            ),
        ),
    ),
)


def _simoncelli(image: mx.array) -> object:
    return simoncelli_response(image, 1)


def _riesz_log(image: mx.array) -> object:
    return riesz_log_responses(
        image, (1.0, 1.0), 1.0, 4.0, ((1, 0),), FourierGrid.DFT,
        FilterPadding.MIRROR, 0.0,
    )


def _riesz_simoncelli(image: mx.array) -> object:
    return riesz_simoncelli_responses(image, 1, ((1, 0),), FourierGrid.DFT)


@_REFUSING
@pytest.mark.parametrize(
    "entry_point",
    [_simoncelli, _riesz_log, _riesz_simoncelli],
    ids=["simoncelli", "riesz-log", "riesz-simoncelli"],
)
def test_low_level_fft_entry_points_refuse_before_fft(
    entry_point: Callable[[mx.array], object],
) -> None:
    # Direct callers must not bypass the disabled status and reach a batched FFT.
    with pytest.raises(ValueError, match=_DISABLED):
        entry_point(mx.zeros((8, 8)))


@_REFUSING
@pytest.mark.parametrize(
    "options",
    [options for _, options in _FFT_SETTINGS],
    ids=[name for name, _ in _FFT_SETTINGS],
)
def test_fft_filters_are_rejected_before_extraction(options: TransformSettings) -> None:
    shape = (8, 8, 8)
    spacing = (1.0, 1.0, 1.0)
    image = np.arange(8**3, dtype=np.float64).reshape(shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[1:-1, 1:-1, 1:-1] = 1

    # A mixed request must fail on its first item: the supported Original image
    # may not be yielded before the disabled filter is refused.
    mixed = dataclasses.replace(
        options, image_types=(ImageType.ORIGINAL, *options.image_types),
    )
    yielded = feature_images(
        mx.array(image, dtype=mx.float32), spacing, mixed,
    )
    with pytest.raises(ValueError, match=_DISABLED):
        next(yielded)

    with pytest.raises(ValueError, match=_DISABLED):
        feature_image_names(shape, spacing, options)

    with pytest.raises(ValueError, match=_DISABLED):
        extract(
            image, mask, settings=Settings(features=("firstorder_Mean",)),
            transforms=options,
        )
