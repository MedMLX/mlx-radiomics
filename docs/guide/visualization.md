# Feature-map figures

Install the optional Matplotlib figure export:

```bash
uv pip install "mlxradiomics[viz]"
```

`mlxradiomics.viz` renders evaluated NumPy maps on the CPU. It does not load MLX
or change feature extraction. The core package does not require Matplotlib.

## Plot and export maps

```python
from mlxradiomics import Settings, extract
from mlxradiomics.viz import plot_feature_maps

# image and labels are aligned 2D or 3D NumPy arrays; spacing is in mm.
spacing = (2.0, 1.0, 1.0)
result = extract(
    image, labels, spacing=spacing, mode="voxel",
    settings=Settings(features=("firstorder_Mean", "glcm_Contrast")),
)
figure = plot_feature_maps(
    result.features, image=image, mask=labels == 1,
    spacing=result.metadata["spacing"], cmap="cividis", histograms=True,
)
figure.savefig("feature-comparison.png", dpi=300)
figure.savefig("feature-comparison.pdf")
figure.savefig("feature-comparison.svg")
```

Each feature has a row with its own labeled colorbar. 3D maps have three
orthogonal slices; 2D maps have one panel. Set `indices=(20, 80, 70)` to choose
the displayed slices of a 3D map. The returned native Matplotlib Figure can be
customized. Plotting creates no GUI windows and does not change Matplotlib's
configured backend. Exports replace existing destinations, following `savefig`
semantics.

Set `histograms=True` to include every finite voxel in the whole ROI in a
histogram. Color limits are calculated per feature from its whole-ROI 2nd and
98th percentiles. Histograms show the limits as dashed lines; values outside
the limits use the endpoint colors.

## Colormaps and scaling

`COLORMAPS` contains `viridis`, `plasma`, `inferno`, `magma`, and `cividis`, the
[perceptually uniform sequential palettes in Matplotlib](https://matplotlib.org/stable/gallery/color/colormap_reference.html).
Append `_r` to any name for a reversed palette, such as `magma_r`.

| Option | Meaning |
| --- | --- |
| `clim=(low, high)` | Explicit color limits; useful for comparing the same feature across cases |
| `percentiles=(0, 100)` | Use the entire finite ROI range instead of the default `(2, 98)` |
| `scale="linear"` | Linear normalization, the default |
| `scale="log"` | Logarithmic normalization; rejects zero or negative ROI values and limits |
| `scale="symlog", linthresh=1.0` | Signed logarithmic normalization with a linear region near zero |
| `image_clim=(-160, 240)` | Explicit grayscale background window, in the image's intensity units |
| `alpha=0.8` | Feature overlay opacity between 0 and 1 |

Background images use their finite 1st and 99th percentiles unless an explicit
window is supplied. These are generic display defaults, not modality-specific
CT/MR/PET windows. Constant maps receive a small nonzero display range.

## Geometry and ROI membership

All maps, the optional background, and the mask must already share the **processed
grid**. A boolean mask is required to exclude background; use `labels == label`,
not an integer label image. If no mask is supplied, every voxel participates,
including exported zeros. Valid zero feature values inside the mask remain
visible; NaN and infinities are transparent and excluded from display statistics.
An empty ROI or a feature with no finite ROI values raises `ValueError`.

Supply the actual extraction ROI after resegmentation or intensity-based
exclusions. Raw labels alone can include voxels whose features were not computed.
Never recover ROI membership from `feature_map != 0`.

`spacing` is in millimeters and NumPy axis order, with a default of 1 mm per axis.
Physical aspect is preserved for each plane. Static figure axes show distance
from the first voxel center along array axes, **not patient coordinates or
anatomical directions**.
Inputs are neither reoriented nor resampled, and matching shapes alone cannot
establish file registration. Use a dedicated image viewer for anatomical
orientation or volume rendering.

When extraction resamples or crops a file, do not overlay maps on the original
source array. Supply an image and ROI on the processed grid instead. NIfTI/NRRD
exports with finite values can be read with `mlxradiomics.imaging.io.read_image`; its `Volume.array`
and `Volume.spacing` use the same NumPy-order convention as this API. Confirm
origin and direction as well as shape and spacing before combining file inputs.
That loader rejects nonfinite data; use a loader that preserves NaN/infinity
when inspecting undefined feature maps.

Plotting borrows the map arrays and computes CPU display statistics. Rendering
many maps at once retains their finite ROI samples for the figure. Select a
subset for large studies.

## Runnable example

```bash
python examples/visualize_feature_maps.py
```

The example writes PNG, PDF, and SVG files under `outputs/viz-demo`. Its synthetic
fixtures exercise the renderer and do not represent patient measurements.
