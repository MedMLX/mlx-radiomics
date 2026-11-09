"""Grouped first-order computation preserves complete, ordered streamed results."""

from collections.abc import Mapping
from dataclasses import replace

import numpy as np
import pytest
from conftest import Conventions, settings_for
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.workflows.engine import extract as extract_original
from mlxradiomics.workflows.extraction import extract

FEATURES = (
    "firstorder_Mean", "glcm_Contrast", "firstorder_Variance", "glcm_JointEntropy",
)


def _input() -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    z, y, x = np.indices((9, 8, 7))
    image = (64 * z + 8 * y + x).astype(np.float64)
    mask = np.ones(image.shape, dtype=np.bool_)
    mask[0] = False
    mask[3:5, 3:5, 3:5] = False
    mask[-1, -1, -1] = False
    return image, mask


@pytest.mark.parametrize("conventions", ["pyradiomics", "ibsi"])
@pytest.mark.parametrize("masked", [True, False])
def test_grouped_stream_keeps_all_features_and_coordinates(
    conventions: Conventions, masked: bool,
) -> None:
    image, mask = _input()
    settings = settings_for(
        conventions,
        features=FEATURES,
        bin_width=16,
        scratch_mib=1,
        masked_kernel=masked,
    )
    names = tuple(f"original_{name}" for name in FEATURES)
    batches: list[tuple[NDArray[np.int32], dict[str, NDArray[np.float32]]]] = []

    def consume(centers: NDArray[np.int32], values: Mapping[str, NDArray[np.float32]]) -> None:
        assert tuple(values) == names
        assert all(value.dtype == np.float32 and value.shape == (len(centers),)
                   for value in values.values())
        # Keep borrowed arrays until extraction finishes: later groups must not overwrite them.
        batches.append((centers, dict(values)))

    streamed = extract_original(image, mask, settings=settings, voxel=True, voxel_sink=consume)
    retained = extract_original(image, mask, settings=replace(settings, scratch_mib=8), voxel=True)
    assert not streamed.features
    np.testing.assert_array_equal(np.concatenate([c for c, _ in batches]), np.argwhere(mask))
    metadata = streamed.metadata
    assert metadata["settings"]["features"] == FEATURES
    assert metadata["feature_validity"] == retained.metadata["feature_validity"]
    for name in names:
        values = np.concatenate([batch[name] for _, batch in batches])
        np.testing.assert_array_equal(values, np.asarray(retained.features[name])[mask])

    # Independent local population statistics anchor values and placement at every center.
    mean: list[float] = []
    variance: list[float] = []
    for center in np.argwhere(mask):
        window = tuple(slice(max(0, int(v) - 1), min(s, int(v) + 2))
                       for v, s in zip(center, image.shape, strict=True))
        local = image[window][mask[window]] if masked else image[window].reshape(-1)
        mean.append(float(np.mean(local)))
        variance.append(float(np.var(local)))
    for name, expected in (("firstorder_Mean", mean), ("firstorder_Variance", variance)):
        values = np.concatenate([batch[f"original_{name}"] for _, batch in batches])
        np.testing.assert_allclose(values, expected, rtol=2e-6, atol=2e-6, err_msg=name)


def test_grouped_derived_stream_keeps_image_prefixes_and_validity() -> None:
    image, mask = _input()
    settings = Settings(features=("firstorder_Mean", "glcm_Contrast"), bin_width=16,
                        scratch_mib=1)
    transforms = TransformSettings(image_types=(ImageType.ORIGINAL, ImageType.SQUARE_ROOT))
    maps: dict[str, NDArray[np.float32]] = {}

    def consume(centers: NDArray[np.int32], values: Mapping[str, NDArray[np.float32]]) -> None:
        prefix = next(iter(values)).split("_", 1)[0]
        assert tuple(values) == tuple(f"{prefix}_{name}" for name in settings.features)
        for name, value in values.items():
            maps.setdefault(name, np.zeros(image.shape, dtype=np.float32))[tuple(centers.T)] = value

    streamed = extract(image, mask, settings=settings, transforms=transforms,
                              mode="voxel", sink=consume)
    retained = extract(image, mask, settings=settings, transforms=transforms, mode="voxel")
    assert set(maps) == {
        f"{prefix}_{name}" for prefix in ("original", "squareroot") for name in settings.features
    }
    assert streamed.metadata["feature_validity"] == retained.metadata["feature_validity"]
    assert streamed.metadata["settings"] == retained.metadata["settings"]
    for name, values in maps.items():
        np.testing.assert_array_equal(values, retained.features[name])


def test_grouped_first_order_retains_source_precision_guard() -> None:
    image, mask = _input()
    image[4, 5, 5] = 2.0**-130
    settings = Settings(features=FEATURES, bin_width=16, scratch_mib=1)
    with pytest.raises(RuntimeError, match=r"nonzero float64 sources below 2\*\*-126"):
        extract_original(image, mask, settings=settings, voxel=True)



def test_sparse_unmasked_maps_keep_global_bins_and_stream_original_centers() -> None:
    image = np.zeros((9, 13), dtype=np.float32)
    image[0, 0] = 1000  # Outside every neighborhood, but still anchors global bins.
    image[5:8, 8:12] = np.arange(1, 13, dtype=np.float32).reshape(3, 4)
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask[6, 9:11] = 1
    settings = Settings(features=("firstorder_Mean", "firstorder_Entropy"),
                        masked_kernel=False, bin_count=4)
    retained = extract(image, mask, settings=settings, mode="voxel")
    mean = retained.features["original_firstorder_Mean"]
    assert isinstance(mean, np.ndarray)
    np.testing.assert_array_equal(mean[mask == 1], [6., 7.])
    np.testing.assert_array_equal(retained.features["original_firstorder_Entropy"],
                                  np.zeros(image.shape))
    delivered: list[tuple[NDArray[np.int32], Mapping[str, NDArray[np.float32]]]] = []

    def sink(centers: NDArray[np.int32], values: Mapping[str, NDArray[np.float32]]) -> None:
        delivered.append((centers.copy(), values))

    streamed = extract(image, mask, settings=settings, mode="voxel", sink=sink)
    np.testing.assert_array_equal(np.concatenate([c for c, _ in delivered]), [[6, 9], [6, 10]])
    for name in settings.features:
        retained_values = retained.features[f"original_{name}"]
        assert isinstance(retained_values, np.ndarray)
        np.testing.assert_array_equal(
            np.concatenate([v[f"original_{name}"] for _, v in delivered]),
            retained_values[mask == 1],
        )
    assert streamed.metadata == retained.metadata
