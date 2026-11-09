"""Export synthetic feature-map fixtures as PNG, PDF and SVG figures.

These fixtures exercise visualization; they are not extracted patient features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Protocol, cast

import numpy as np


class _FigureWriter(Protocol):
    def savefig(self, path: Path, *, dpi: int) -> None: ...


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/viz-demo"))
    args = parser.parse_args()
    from mlxradiomics.viz import plot_feature_maps

    z, y, x = np.meshgrid(
        np.linspace(-1, 1, 40, dtype=np.float64),
        np.linspace(-1, 1, 80, dtype=np.float64),
        np.linspace(-1, 1, 96, dtype=np.float64),
        indexing="ij",
    )
    radius = (x / 0.75) ** 2 + (y / 0.8) ** 2 + (z / 0.75) ** 2
    image = np.where(radius < 1, 50 + 30 * np.cos(y * 9), -200)
    mask = ((x + 0.1) / 0.42) ** 2 + (y / 0.45) ** 2 + (z / 0.5) ** 2 < 1
    maps = {"Synthetic fixture · spatial gradient": np.where(mask, 50 + 40 * x + 20 * z, 0)}
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure = plot_feature_maps(
        maps,
        image=image,
        mask=mask,
        spacing=(2, 1, 1),
        cmap="cividis",
        histograms=True,
    )
    for suffix in ("png", "pdf", "svg"):
        cast(_FigureWriter, figure).savefig(output_dir / f"synthetic-feature.{suffix}", dpi=200)


if __name__ == "__main__":
    main()
