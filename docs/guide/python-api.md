# Python API

Four functions cover arrays and files, in memory and on disk. They share one
settings and result contract.

| Operation | Arrays | NIfTI/NRRD files | Result |
| --- | --- | --- | --- |
| ROI features | `extract(image, mask)` | `extract_files(image_path, mask_path)` | `Result[float]` |
| Voxel maps in memory | `extract(image, mask, mode="voxel")` | `extract_files(image_path, mask_path, mode="voxel")` | `Result[NDArray[np.float64]]` |
| Voxel maps on disk | `export(image, mask, output_dir)` | `export_files(image_path, mask_path, output_dir)` | `MapExport(report, maps, metadata)` |

All options are keyword-only.

- Every operation accepts `settings=` ({class}`~mlxradiomics.Settings`) and
  `transforms=` ({class}`~mlxradiomics.TransformSettings`).
- Array operations accept `spacing=` in millimeters, in NumPy axis order.
- File operations read spacing from the image. They accept `image_settings=`
  ({class}`~mlxradiomics.ImageSettings`) for resampling and mask correction.
- Array `export` accepts `allow_synthetic_geometry=` to acknowledge its synthetic grid.
- Exports accept `overwrite=`, `temporary_dir=` and `max_temporary_mib=`.

Settings, enums and parameter-file parsing load without MLX or SimpleITK. The
execution runtime loads when extraction or export is imported.

The optional [visualization API](visualization.md) provides static feature-map
figures, ROI overlays, perceptually uniform colormaps, and PNG/PDF/SVG export
through `mlxradiomics.viz`.

## Example

```python
from mlxradiomics import Settings, extract, extract_files, export_files

settings = Settings(label=2, features=("firstorder_Mean", "glcm_Contrast"))

roi = extract(image, mask, spacing=(3.0, 1.0, 1.0), settings=settings)
mean = roi.features["original_firstorder_Mean"]  # float

voxel = extract_files("image.nii.gz", "mask.nii.gz", settings=settings, mode="voxel")
mean_map = voxel.features["original_firstorder_Mean"]  # float64 NumPy array

saved = export_files("image.nii.gz", "mask.nii.gz", "outputs/maps", settings=settings)
saved.report                            # pathlib.Path to features.json
saved.maps["original_firstorder_Mean"]  # pathlib.Path to a float32 NIfTI
saved.metadata["settings"]              # effective extraction settings
```

Extraction returns `.features` and `.metadata`. Exports return `.report`, `.maps`
and `.metadata`. Effective transforms are in `.metadata["transforms"]`. `mode`
selects the value type for static type checkers as well as at runtime. In-memory
maps are float64 NumPy arrays holding float32 results, zero outside the ROI.
Saved NIfTI maps are float32. The float64 container does not increase calculation
precision. Use export or streaming for large images instead of retaining all maps.

## Geometry

File operations preserve the processed image's physical geometry. For arrays,
pass `spacing` in NumPy axis order: `(z, y, x)` for a conventional 3-D image
loaded with SimpleITK. Omitted spacing warns and assumes 1 mm on every axis.
Even with spacing supplied, array `export` has no source origin or direction;
it writes zero origin and identity direction and warns. Use `export_files` for
scanner-aligned maps. For deliberately synthetic arrays, pass
`allow_synthetic_geometry=True` to acknowledge that coordinate system.

## Feature selection

```python
from mlxradiomics import Settings, list_features

settings = Settings(features=list_features("firstorder", "glcm"))
voxel_names = list_features(mode="voxel", ndim=3)
```

`list_features` returns catalogue names filtered by family, mode and optional
dimension; it does not change the 93-feature default. Unknown family names raise
`ValueError`. Shape and local intensity are excluded from voxel selections;
3-D discovery excludes `shape2D` because its 3-D use requires a single slice and
explicit `force2d_axis`. Inspect the returned names before using a large selection.

`export` and `export_files` always write voxel maps, not ROI scalar reports.
Use `mlxradiomics.medmlx.run` for ROI JSON/CSV, or consume `Result.features` in
Python.

## Derived images

`TransformSettings.image_types` selects the images that features are computed on.
Each derived image is discretized separately, and its features carry its name as a
prefix:

```python
from mlxradiomics import ImageType, Settings, TransformSettings, export_files
from mlxradiomics.config.transforms import WaveletSettings

result = export_files(
    "image.nii.gz", "mask.nii.gz", "outputs/maps",
    settings=Settings(features=("firstorder_Mean", "glcm_Contrast")),
    transforms=TransformSettings(
        image_types=(ImageType.ORIGINAL, ImageType.WAVELET),
        wavelet=WaveletSettings(level=1),
    ),
)
# result.maps keys: original_glcm_Contrast, wavelet-LLH_glcm_Contrast, ...
```

Each filter has its own settings type in {mod}`mlxradiomics.config.transforms`,
validated when it is constructed:

```python
from mlxradiomics.config.transforms import LoGSettings, MeanSettings

transforms = TransformSettings(
    image_types=(ImageType.ORIGINAL, ImageType.MEAN, ImageType.LOG),
    mean=MeanSettings(support=5),
    log=LoGSettings(sigma=(1.0, 2.0)),
)
```

In ROI mode, shape features are computed once from the original inputs. Normalization and
resegmentation run once, before any derived image is computed. See
{mod}`mlxradiomics.config.transforms` for each filter's options,
{mod}`mlxradiomics.config.filtering` for padding, pooling and response enums, and
{class}`~mlxradiomics.ImageType` for the available image types.

## Conventions

Each calculation convention is its own `Settings` field with a short list of
named values. Defaults follow IBSI; select an alternative explicitly for the
intended scientific method:

| Field | IBSI default | Alternative | Controls |
| --- | --- | --- | --- |
| `covariance_divisor` | `"n-1"` | `"n"` | Divisor of the voxel-center covariance behind the shape axes and the approximate enclosing ellipsoid |
| `ellipsoid_area` | `"ibsi-series"` | `"exact"` | Surface area of the approximate enclosing ellipsoid in its area density |
| `kurtosis` | `"excess"` | `"raw"` | First-order kurtosis with or without subtracting three |
| `zero_variance_moments` | `"zero"` | `"nan"` | Skewness and kurtosis of a zero-variance ROI |
| `ngtdm_singular_cases` | `"ibsi-caps"` | `"unadjusted"` | Coarseness capped at 1e6, busyness 0 with one occupied level, strength 0 for zero differences |
| `neighborhood_support` | `"closed-radius"` | `"distances"` | NGLDM/NGTDM neighborhood: every voxel within `neighborhood_radius`, or the shells in `distances` |
| `bin_origin` | `"resegment-or-minimum"` | `"zero-lattice"` | Origin of a fixed-bin-size lattice when `bin_anchor` is `None` |
| `constant_roi_bin` | `"first"` | `"middle"` | Gray level of a constant ROI under `bin_count` |
| `mask_extrapolation` | `"zero"` | `"nearest"` | Mask outside the outermost source voxel centers under IBSI-grid interpolation (`zero` follows MIRP) |
| `spline_boundary` | `"nearest"` | `"mirror"` | Boundary of the cubic B-spline image interpolator under IBSI-grid interpolation: edge replication (as scipy `mode="nearest"` in MIRP) or ITK's mirror condition |

For example, `Settings(bin_width=25.0, kurtosis="raw", covariance_divisor="n")`
selects the raw fourth-moment ratio and population covariance while retaining
the other defaults. The matching parameter-file keys are `covarianceDivisor`,
`ellipsoidArea`, `kurtosis`, `zeroVarianceMoments`, `ngtdmSingularCases`,
`neighborhoodSupport`, `binOrigin`, `constantRoiBin`, `maskExtrapolation` and
`splineBoundary`. Conventions that have no alternative are not settings: the gray-level
count of normalized inverse difference and moment is the number of
discretization bins, histogram mode ties choose the bin closest to the mean and
then the lower bin, and the Simoncelli/Riesz frequency grid is the `grid`
argument of the filter functions (DFT by default).

Per-family `*_aggregation` fields choose how texture matrices are combined
across directions and slices; see {class}`~mlxradiomics.Settings` for their
accepted values. The [feature catalogue](features.md#catalogue) maps public
names to IBSI identifiers.

## Streaming voxel batches

To consume voxel results as they are computed, pass `mode="voxel", sink=consume`
to either extraction function:

```python
def consume(centers, features):
    # centers: int32 (batch, ndim) in NumPy axis order
    # features: name -> float32 (batch,)
    ...

result = extract(image, mask, mode="voxel", sink=consume)
```

The arrays are borrowed for the duration of the call, so copy any you keep. The
returned `.features` is empty; `.metadata` still holds validity counts. To own
resources per derived image, pass `sink_factory=`: a callable that takes the image
name and returns a context manager yielding a sink. Supply at most one of `sink`
and `sink_factory`.

## Multi-case studies

{func}`~mlxradiomics.workflows.study.run_study` runs every case and label
combination, recording input, I/O and extraction failures before continuing:

```python
from mlxradiomics.workflows.study import run_study

outcomes = run_study(
    [{"case_id": "case-a", "image_path": "a.nii.gz", "mask_path": "a-mask.nii.gz"}],
    "outputs/study",
    labels=(1, 2),
    params={"featureClass": {"firstorder": ["Mean"], "glcm": ["Contrast"]}},
)
failed = [o for o in outcomes if o["status"] == "failed"]
```

Outputs go to `<output_dir>/<case_id>/label-<label>`. Unexpected programming
errors propagate immediately, with staging cleaned and previous outputs preserved.

## Errors

Invalid inputs and unsupported options in `extract` and `extract_files` raise
`ValueError`; file-read failures can raise `RuntimeError` or `OSError`.
`TransformSettings` and `ImageSettings` validate on construction; `Settings`
validates when extraction starts. For early settings checks, call `settings.validate(ndim=3)` before loading data.
`bin_count` takes precedence over `bin_width`; set only the discretizer you intend.
The MedMLX runner and exports raise
`InvalidInputError` and `ModelExecutionError` instead; see
[MedMLX runner](runner.md#errors).
