"""Typed subset of Matplotlib's drawing API.

Matplotlib 3.11's shipped stubs leave artist keyword arguments and object-array
returns untyped. These protocols describe only calls used here, verified against
the installed ``axes``, ``figure``, and ``backend_bases`` signatures.
"""

import os
from typing import BinaryIO, Literal, Protocol, cast

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.figure import Figure, SubFigure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.text import Text
from numpy.typing import NDArray

from mlxradiomics.contracts.types import HostArray


class AxesDrawing(Protocol):
    def imshow(
        self,
        values: HostArray,
        *,
        cmap: str | Colormap,
        norm: Normalize | None = None,
        alpha: float = 1,
        origin: Literal["upper"],
        extent: tuple[float, float, float, float],
        interpolation: Literal["nearest"],
        aspect: Literal["equal"],
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> AxesImage: ...
    def set_xlabel(self, label: str) -> Text: ...
    def set_ylabel(self, label: str) -> Text: ...
    def set_title(self, label: str, *, fontsize: float) -> Text: ...
    def bar(
        self,
        x: HostArray,
        height: HostArray,
        *,
        width: HostArray,
        color: HostArray,
        linewidth: float,
    ) -> object: ...
    def axvline(self, x: float, *, color: str, linestyle: str, linewidth: float) -> Line2D: ...
    def set_xscale(self, value: str, *, linthresh: float = 1) -> None: ...


class FigureDrawing(Protocol):
    def suptitle(self, text: str, *, fontsize: float) -> Text: ...
    def colorbar(
        self,
        artist: AxesImage,
        *,
        ax: list[Axes],
        label: str,
        shrink: float,
        extend: Literal["both"],
    ) -> Colorbar: ...
    def subfigures(self, *, nrows: int, squeeze: Literal[False]) -> NDArray[np.object_]: ...
    def subplots(self, *, ncols: int, squeeze: Literal[False]) -> NDArray[np.object_]: ...
    def savefig(self, path: str | os.PathLike[str] | BinaryIO, *, dpi: int = 200) -> None: ...


def new_figure(*, figsize: tuple[float, float]) -> Figure:
    """Create a pyplot-free Figure with constrained layout."""
    return Figure(figsize=figsize, layout="constrained")


def axes_drawing(axes: Axes) -> AxesDrawing:
    return cast(AxesDrawing, axes)


def figure_drawing(figure: Figure | SubFigure) -> FigureDrawing:
    return cast(FigureDrawing, figure)
