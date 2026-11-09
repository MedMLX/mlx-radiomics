"""Feature-map figures must preserve ROI membership, physical geometry and scaling."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.image import AxesImage
from PIL import Image

from mlxradiomics.viz import plot_feature_maps
from mlxradiomics.viz._backend import figure_drawing

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def overlays(axes: list[Axes]) -> list[AxesImage]:
    return [ax.images[-1] for ax in axes if ax.images]


def test_masks_valid_zeros_nonfinite_and_physical_aspect_survive_export(tmp_path: Path) -> None:
    values = np.array([[0, 1, 500], [np.nan, np.inf, 2]], dtype=np.float64)
    mask = np.array([[True, True, False], [True, True, True]], dtype=np.bool_)
    figure = plot_feature_maps(
        {"Zero is a valid feature": values},
        mask=mask,
        spacing=(3, 2),
        cmap="viridis",
        alpha=1,
        percentiles=(0, 100),
        histograms=True,
    )
    artist = overlays(figure.axes)[0]
    pixels = artist.get_array()
    assert pixels is not None
    np.testing.assert_array_equal(
        np.ma.getmaskarray(pixels),
        [[False, False, True], [True, True, False]],
    )
    assert artist.get_extent() == [-1, 5, 4.5, -1.5]
    assert artist.get_clim() == (0, 2)
    assert artist.axes is not None
    assert artist.axes.get_aspect() == 1
    histogram = next(ax for ax in figure.axes if ax.get_ylabel() == "Voxels")
    assert histogram.get_title() == "Finite ROI · n=3"
    png = tmp_path / "masked.png"
    figure_drawing(figure).savefig(png, dpi=100)
    with Image.open(png) as raster:
        # Center of the valid zero must contain viridis purple, not the panel background.
        x, y = artist.axes.transData.transform((0, 0))
        pixel = raster.convert("RGB").getpixel((round(x), raster.height - round(y)))
        assert pixel == (68, 1, 84)


def test_constant_signed_and_single_slice_maps_have_defined_scales() -> None:
    constant = plot_feature_maps({"constant": np.zeros((1, 2, 3), dtype=np.float32)})
    assert len(overlays(constant.axes)) == 3
    for artist in overlays(constant.axes):
        assert artist.get_clim() == (-0.5, 0.5)
    signed = np.array([[-100, -1], [0, 100]], dtype=np.float64)
    figure = plot_feature_maps({"signed": signed}, scale="symlog", clim=(-100, 100))
    norm = overlays(figure.axes)[0].norm
    np.testing.assert_array_equal(norm(np.array([-100, 0, 100], dtype=np.float64)), [0, 0.5, 1])
    positive = plot_feature_maps(
        {"positive": np.array([[1, 10], [10, 100]], dtype=np.float64)}, scale="log", clim=(1, 100)
    )
    np.testing.assert_array_equal(
        overlays(positive.axes)[0].norm(np.array([10], dtype=np.float64)),
        [0.5],
    )


def test_invalid_grids_and_log_values_fail_before_creating_figures() -> None:
    values = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="boolean array"):
        plot_feature_maps({"map": values}, mask=np.ones((2, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="on the feature grid"):
        plot_feature_maps({"map": values}, image=np.zeros((3, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="positive finite"):
        plot_feature_maps({"map": values}, spacing=(0, 1))
    with pytest.raises(ValueError, match="no finite values"):
        plot_feature_maps({"map": np.full((2, 3), np.nan, dtype=np.float64)})
    with pytest.raises(ValueError, match="strictly positive ROI"):
        plot_feature_maps({"map": values}, scale="log")
    with pytest.raises(ValueError, match="in-bounds integer"):
        plot_feature_maps({"map": values}, indices=cast(tuple[int, ...], (0.5, 1)))
    assert plt.get_fignums() == []

