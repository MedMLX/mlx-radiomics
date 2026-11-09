"""Spacing-aware feature panels using NumPy and Matplotlib on the CPU."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from textwrap import fill
from typing import Literal, cast

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from numpy.typing import NDArray

from mlxradiomics.contracts.types import HostArray
from mlxradiomics.viz._backend import axes_drawing, figure_drawing, new_figure

__all__ = ["COLORMAPS", "plot_feature_maps"]

COLORMAPS = ("viridis", "plasma", "inferno", "magma", "cividis")
"""Perceptually uniform sequential palettes; append ``_r`` to reverse one."""

type Scale = Literal["linear", "log", "symlog"]
type Limits = tuple[float, float]


@dataclass(frozen=True)
class MapData:
    maps: Mapping[str, HostArray]
    mask: NDArray[np.bool_]
    image: HostArray | None
    spacing: tuple[float, ...]
    indices: tuple[int, ...]

    @property
    def planes(self) -> tuple[int | None, ...]:
        return (None,) if self.mask.ndim == 2 else (0, 1, 2)

    def values(self, name: str) -> NDArray[np.float64]:
        array = self.maps[name]
        values = np.asarray(array[self.mask & np.isfinite(array)], dtype=np.float64)
        if not values.size:
            raise ValueError(f"Feature {name!r} has no finite values inside the mask.")
        return values


def prepare_maps(
    maps: Mapping[str, HostArray],
    mask: HostArray | None,
    image: HostArray | None,
    spacing: tuple[float, ...] | None,
    indices: tuple[int, ...] | None,
) -> MapData:
    if not maps:
        raise ValueError("maps must contain at least one 2D or 3D feature map.")
    arrays = dict(maps)
    if any(not isinstance(cast(object, array), np.ndarray) for array in arrays.values()):
        raise ValueError("maps must contain NumPy arrays; extract with mode='voxel'.")
    shape = next(iter(arrays.values())).shape
    if len(shape) not in (2, 3) or any(size == 0 for size in shape):
        raise ValueError("maps must contain nonempty 2D or 3D arrays.")
    for name, array in arrays.items():
        if not isinstance(cast(object, name), str) or not name.strip():
            raise ValueError("Feature names must be nonempty strings.")
        if array.shape != shape or array.dtype.kind not in "iuf":
            raise ValueError(f"Feature {name!r} must be a real numeric array of shape {shape}.")
    if mask is not None and (mask.shape != shape or mask.dtype.kind != "b"):
        raise ValueError("mask must be a boolean array on the feature grid; use labels == label.")
    roi = np.ones(shape, dtype=np.bool_) if mask is None else np.asarray(mask, dtype=np.bool_)
    if not np.any(roi):
        raise ValueError("mask must include at least one voxel.")
    if image is not None and (image.shape != shape or image.dtype.kind not in "iuf"):
        raise ValueError("image must be a real numeric array on the feature grid.")
    steps = (1.0,) * len(shape) if spacing is None else tuple(spacing)
    if len(steps) != len(shape) or any(not np.isfinite(x) or x <= 0 for x in steps):
        raise ValueError("spacing must have one positive finite value per NumPy axis.")
    if indices is None:
        centers: list[int] = []
        for axis in range(roi.ndim):
            other_axes = tuple(i for i in range(roi.ndim) if i != axis)
            positions = np.flatnonzero(np.any(roi, axis=other_axes))
            centers.append(int((positions[0] + positions[-1]) // 2))
        indices = tuple(centers)
    validate_indices(indices, shape)
    return MapData(arrays, roi, image, steps, indices)


def validate_indices(indices: tuple[int, ...], shape: tuple[int, ...]) -> None:
    if len(indices) != len(shape) or any(
        isinstance(index, bool) or not isinstance(cast(object, index), (int, np.integer))
        or not 0 <= index < size for index, size in zip(indices, shape, strict=True)
    ):
        raise ValueError("indices must contain one in-bounds integer per NumPy axis.")


def get_colormap(name: str) -> Colormap:
    if name.removesuffix("_r") not in COLORMAPS:
        raise ValueError(f"cmap must be one of {COLORMAPS}, optionally suffixed with '_r'.")
    return mpl.colormaps[name].with_extremes(bad=(0, 0, 0, 0))


def validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1.")


def validate_limits(limits: Limits, argument: str = "clim") -> Limits:
    if len(limits) != 2 or not all(np.isfinite(x) for x in limits) or limits[0] >= limits[1]:
        raise ValueError(f"{argument} must contain two finite increasing values.")
    return float(limits[0]), float(limits[1])


def make_norm(
    values: NDArray[np.float64], scale: Scale, clim: Limits | None,
    percentiles: Limits, linthresh: float,
) -> Normalize:
    if scale not in ("linear", "log", "symlog"):
        raise ValueError("scale must be 'linear', 'log', or 'symlog'.")
    low, high = validate_limits(percentiles, "percentiles")
    if low < 0 or high > 100:
        raise ValueError("percentiles must lie between 0 and 100.")
    if not np.isfinite(linthresh) or linthresh <= 0:
        raise ValueError("linthresh must be positive and finite.")
    if scale == "log" and np.any(values <= 0):
        raise ValueError("log scale requires strictly positive ROI values; use linear or symlog.")
    if clim is None:
        vmin, vmax = (float(x) for x in np.percentile(values, (low, high)))
        if vmin == vmax:
            vmin, vmax = float(values.min()), float(values.max())
        if vmin == vmax:
            if scale == "log":
                vmin, vmax = vmin / 1.01, vmax * 1.01
            else:
                padding = max(abs(vmin) * 0.01, 0.5)
                vmin, vmax = vmin - padding, vmax + padding
    else:
        vmin, vmax = validate_limits(clim)
    if scale == "log":
        if vmin <= 0:
            raise ValueError("log scale requires strictly positive color limits.")
        return LogNorm(vmin=vmin, vmax=vmax)
    if scale == "symlog":
        return SymLogNorm(linthresh, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def image_limits(data: MapData, clim: Limits | None) -> Limits | None:
    if clim is not None:
        return validate_limits(clim, "image_clim")
    if data.image is None:
        return None
    finite = data.image[np.isfinite(data.image)]
    if not finite.size:
        raise ValueError("image must contain finite intensities.")
    norm = make_norm(np.asarray(finite, dtype=np.float64), "linear", None, (1, 99), 1)
    return norm_limits(norm)


def norm_limits(norm: Normalize) -> Limits:
    assert norm.vmin is not None and norm.vmax is not None
    return norm.vmin, norm.vmax


def plane_data[T: np.generic](
    array: NDArray[T], axis: int | None, indices: tuple[int, ...],
) -> NDArray[T]:
    return array if axis is None else np.take(array, indices[axis], axis=axis)


def draw_plane(
    ax: Axes, data: MapData, name: str, axis: int | None,
    indices: tuple[int, ...], cmap: Colormap, norm: Normalize, alpha: float,
    background_clim: Limits | None,
) -> AxesImage:
    remaining = [i for i in range(data.mask.ndim) if i != axis]
    row, column = remaining
    height, width = data.mask.shape[row], data.mask.shape[column]
    dy, dx = data.spacing[row], data.spacing[column]
    extent = (-dx / 2, (width - 0.5) * dx, (height - 0.5) * dy, -dy / 2)
    ax.set_facecolor("#20242b")
    drawing = axes_drawing(ax)
    if data.image is not None:
        assert background_clim is not None
        drawing.imshow(
            plane_data(data.image, axis, indices), cmap="gray", origin="upper",
            extent=extent, vmin=background_clim[0], vmax=background_clim[1],
            interpolation="nearest", aspect="equal",
        )
    values = plane_data(data.maps[name], axis, indices)
    roi = plane_data(data.mask, axis, indices)
    artist = drawing.imshow(
        np.ma.array(values, mask=~roi | ~np.isfinite(values)), cmap=cmap, norm=norm,
        alpha=alpha, origin="upper", extent=extent, interpolation="nearest", aspect="equal",
    )
    drawing.set_xlabel(f"Array axis {column} (mm)")
    drawing.set_ylabel(f"Array axis {row} (mm)")
    drawing.set_title("2D" if axis is None else f"Axis {axis} · slice {indices[axis]}", fontsize=10)
    return artist


def draw_histogram(ax: Axes, values: NDArray[np.float64], norm: Normalize, cmap: Colormap) -> None:
    counts, edges = np.histogram(values, bins=48)
    centers = (edges[:-1] + edges[1:]) / 2
    drawing = axes_drawing(ax)
    drawing.bar(centers, counts, width=np.diff(edges), color=cmap(norm(centers)), linewidth=0)
    low, high = norm_limits(norm)
    drawing.axvline(low, color="#555555", linestyle="--", linewidth=1)
    drawing.axvline(high, color="#555555", linestyle="--", linewidth=1)
    drawing.set_xlabel("Feature value")
    drawing.set_ylabel("Voxels")
    drawing.set_title(f"Finite ROI · n={values.size:,}", fontsize=10)
    if isinstance(norm, LogNorm):
        drawing.set_xscale("log")
    elif isinstance(norm, SymLogNorm):
        drawing.set_xscale("symlog", linthresh=norm.linthresh)


def plot_feature_maps(
    maps: Mapping[str, HostArray], *, mask: HostArray | None = None,
    image: HostArray | None = None, spacing: tuple[float, ...] | None = None,
    indices: tuple[int, ...] | None = None, cmap: str = "viridis", alpha: float = 0.8,
    clim: Limits | None = None, percentiles: Limits = (2, 98),
    scale: Scale = "linear", linthresh: float = 1.0, image_clim: Limits | None = None,
    histograms: bool = False,
) -> Figure:
    """Plot feature-map comparisons with physical aspect and labeled colorbars.

    Each mapping entry gets a row: one panel for 2D, three array-axis slices for
    3D, plus an optional whole-ROI histogram. Inputs must already share a grid;
    no resampling or anatomical reorientation is performed. NumPy/Matplotlib
    rendering and display statistics execute on the CPU, without loading MLX.

    Args:
        maps: Nonempty mapping of names to aligned real 2D/3D NumPy arrays.
            Accepts ``Result.features`` from voxel extraction.
        mask: Boolean ROI on the processed grid; use ``labels == label``. If
            omitted every voxel participates, including exported zero background.
            Valid zero values inside the mask are always displayed.
        image: Optional aligned grayscale background image.
        spacing: Millimeters per NumPy axis, default 1 mm on each axis.
        indices: One slice index per axis; defaults to the ROI bounding-box center.
        cmap: One of ``COLORMAPS`` or its reversed ``_r`` variant.
        alpha: Overlay opacity in [0, 1].
        clim: Fixed feature color limits, shared across all rows when supplied.
        percentiles: Whole-ROI limits used per feature when ``clim`` is absent.
        scale: Linear, positive-only logarithmic, or signed symmetric logarithmic.
        linthresh: Positive linear-region threshold for symmetric logarithmic scaling.
        image_clim: Background intensity window; defaults to its 1st/99th percentiles.
        histograms: Add a finite whole-ROI distribution to each row.

    Returns:
        A Matplotlib Figure; customize it or use ``savefig`` for PNG/PDF/SVG.

    Raises:
        ValueError: Grids, ROI, display settings, or finite feature values are invalid.
    """
    data = prepare_maps(maps, mask, image, spacing, indices)
    palette = get_colormap(cmap)
    validate_alpha(alpha)
    background_clim = image_limits(data, image_clim)
    values = {name: data.values(name) for name in data.maps}
    norms = {
        name: make_norm(sample, scale, clim, percentiles, linthresh)
        for name, sample in values.items()
    }
    columns = len(data.planes) + int(histograms)
    figure = new_figure(figsize=(4.2 * columns, 4.1 * len(data.maps)))
    rows = figure_drawing(figure).subfigures(nrows=len(data.maps), squeeze=False)
    for row, (name, norm) in enumerate(norms.items()):
        subfigure = figure_drawing(cast(SubFigure, rows[row, 0]))
        subfigure.suptitle(fill(name, width=75), fontsize=11)
        axes = [cast(Axes, ax) for ax in subfigure.subplots(ncols=columns, squeeze=False)[0]]
        plane_axes = axes[:len(data.planes)]
        artists = [
            draw_plane(ax, data, name, axis, data.indices, palette, norm, alpha, background_clim)
            for ax, axis in zip(plane_axes, data.planes, strict=True)
        ]
        subfigure.colorbar(
            artists[0], ax=plane_axes, label="Feature value", shrink=0.8, extend="both",
        )
        if histograms:
            draw_histogram(axes[-1], values[name], norm, palette)
    return figure
