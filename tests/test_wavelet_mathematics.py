"""Stationary-wavelet dilation, phase and boundary behavior against independent filters."""

import numpy as np
import pytest
import pywt
from numpy.typing import NDArray
from radiomics_reference.real_images import hcc_crop

from mlxradiomics.config.transforms import ImageType, TransformSettings, WaveletSettings
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import mx


def test_level_two_haar_uses_dilated_filter() -> None:
    ct, _, _ = hcc_crop(np.s_[105:106, 215:219, 64])
    image = ct.astype(np.float32)
    options = TransformSettings(
        image_types=(ImageType.WAVELET,), wavelet=WaveletSettings(name="haar", level=2)
    )
    with mx.stream(mx.gpu):
        actual = {item.name: np.asarray(item.image, dtype=np.float32) for item in feature_images(
            mx.array(image), (1.0, 1.0), options, force2d_axis=0,
        )}
    # The second-level taps are two samples apart; PyWavelets supplies the
    # independent stationary transform, including phase and periodic boundaries.
    expected = pywt.swtn(ct, "haar", 2, axes=(1,))[0]["d"]
    np.testing.assert_allclose(actual["wavelet2-H"], expected,
                               rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("wavelet", ["haar", "db3", "coif1"])
@pytest.mark.parametrize("start", [0, 1])
def test_multilevel_wavelet_matches_stationary_transform(wavelet: str, start: int) -> None:
    image, _, _ = hcc_crop(np.s_[105:110, 215:222, 62:71])
    levels = 2
    multiple = 2 ** (levels + start)
    padded = np.pad(image, [(0, 0), (0, -7 % multiple), (0, -9 % multiple)], mode="wrap")
    # Transform axes in output-code order, preserving the unfiltered first axis.
    expected = list(reversed(pywt.swtn(padded, wavelet, levels + start, axes=(2, 1))))
    options = TransformSettings(
        image_types=(ImageType.WAVELET,),
        wavelet=WaveletSettings(name=wavelet, level=levels, start_level=start),
    )
    with mx.stream(mx.gpu):
        actual = {item.name: np.asarray(item.image, dtype=np.float32) for item in feature_images(
            mx.array(image.astype(np.float32)), (1.0, 1.0, 1.0), options, force2d_axis=0,
        )}
    targets: dict[str, NDArray[np.float64]] = {}
    for output_level, coefficients in enumerate(expected[start:], 1):
        prefix = "wavelet" if output_level == 1 else f"wavelet{output_level}"
        for code, values in coefficients.items():
            if code == "aa" and output_level != levels:
                continue
            name = code.replace("a", "L").replace("d", "H")
            targets[f"{prefix}-{name}"] = values[:, :7, :9]
    assert set(actual) == set(targets)
    for name, values in targets.items():
        np.testing.assert_allclose(actual[name], values, rtol=1e-4, atol=1e-4, err_msg=name)
