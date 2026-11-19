# Release notes

## Unreleased

### Fixes

**Incomplete and damaged NIfTI files are refused.** A NIfTI image or mask that ends
before the voxel data its header declares, or whose gzip stream is cut off or fails its
checksum, now raises `ValueError`. Such files were read without an error: a mask cut off
half-way loaded with the missing part empty, so features described a smaller ROI.
Uncompressed NIfTI and raw NRRD carry no checksum, so a changed byte that keeps the
file length is still not detectable there.

**An empty `featureClass` list is refused.** `firstorder: []` in a parameter file
dropped the whole class without an error, while PyRadiomics reads it as every feature
of the class. It now raises like a missing list: name the features explicitly.

**Repeated keys in a parameter file are refused.** A second `setting:` section or a
repeated `binWidth` silently replaced the first. The error names the key and its line.

**NumPy numbers in `Settings` get an accurate error.** `bin_width=np.float64(25)` or
`label=np.int64(1)` reported a range error such as "must be finite and positive". The
error now names the field and asks for a Python number.

**Scratch errors name the 1.1.0 options.** A `scratch_dir` that is not a directory and a
voxel export over the `max_scratch_mib` cap now report `scratch_dir` and
`max_scratch_mib` instead of the retired `temporary_dir` and "temporary budget". A
`scratch_dir` that does not exist raises `InvalidInputError` naming the option.

**NGTDM `Contrast` no longer overflows from 46,342 occupied gray levels.** The
level-pair count `Ng (Ng - 1)` wrapped in int32, and the feature came out about two
billion times too large (3.58e8 for 0.1667). Only very fine binning reaches that
many levels.

**Texture variances square exactly.** `GrayLevelVariance`, `RunVariance`,
`ZoneVariance`, `DependenceVariance`, `ZoneDistanceVariance` and the fine-binned
GLCM `DifferenceVariance` squared deviations with a float32 power that some Apple
GPUs do not round correctly; they now multiply. On an M1 Max these features change
by at most 3.3e-7 relative in ROI values and voxel maps. No other feature changes.

**A voxel-map export raises when a scratch write fails.** A write that failed
part-way, as on a full disk, was not reported. If space returned during the run,
the exported maps held values at wrong voxels; otherwise the export failed at the
end with an unrelated shape error. The failed write now raises `OSError`.

**`run_study` continues past an ROI over the working-memory budget.** An ROI count
buffer above `scratch_mib` raised `MemoryError` out of `run_study`, which ended the
study and lost the outcomes of the cases already finished. The case is now recorded
as failed, and `run` raises `ModelExecutionError`.

**Undefined `CentreOfMassShift` and enclosing-ellipsoid densities are NaN.**
`shape_CentreOfMassShift` raised, ending the whole extraction, when the ROI
intensities summed to exactly zero. The approximate enclosing ellipsoid volume and
area densities returned `+inf` for single-slice and other rank-deficient ROIs. Both
are undefined values: they are now NaN, counted as `nan` in `feature_validity`, and
the other features of the extraction are computed as before.

**LBP3D accepts ordinary integer CT and MR ROIs.** It raised "rounding threshold is
ambiguous" on most CT ROIs above about a thousand voxels. Interpolated samples
within float32 uncertainty of a rounding threshold, bounded from the spline
coefficients in each sample's support, are now evaluated in float64 on the CPU with
SciPy and rounded as PyRadiomics does: about 2% of samples on a CT liver and tumor.
Codes, band norms and kurtosis match PyRadiomics' arithmetic on the liver, the
tumor and 28 cubes from liver CT, lung CT and hippocampus MR. On ROIs that were
accepted before, kurtosis moves by up to 2e-3 at a few voxels, toward the
reference. The 715,158-voxel liver takes under 1 s on an M1 Max with 61 MB of host
memory for the float64 coefficients. An exact tie in float64 still raises
`ValueError`, and floating-point images keep their previous check.

**NumPy scalars in spacing and mask threshold are refused before extraction.**
`ImageSettings.resampled_spacing`, `IbsiInterpolation.spacing` and `mask_threshold`,
and the `spacing` argument of `extract`, accepted values such as `np.float32`. The
extraction ran, and writing the run report then failed with `cannot serialize value
of type float32`. They now raise `ValueError` naming the field; Python numbers and
`np.float64` work as before.

**The IVH error for a non-integer range bound says what is required.** IVH on
integer intensities with a non-integer absolute resegmentation bound and no
`ivh_bin_width` or `ivh_bin_count` called the intensities continuous. The message
now asks for integer intensities and integer range bounds.

### Lower memory

Figures are from an M1 Max and a 512×512×99 CT with a 715,158-voxel liver mask.
Feature values are unchanged apart from the variances above.

**Texture matrices of a large ROI.** GLSZM, GLDZM and fine-binned GLCM sorted every
voxel of the ROI window instead of the cells that are counted. The default features
of the liver peak at 153 MiB of MLX memory instead of 416 MiB.

**Recursive LoG.** Each pass writes two image-sized buffers instead of five and
filters along its axis without a transposed copy. LoG at 3 mm peaks at 693 MB of MLX
memory instead of 2,079 MB and takes 146 ms instead of 517 ms.

**Laws with orientation pooling.** Rotated responses are pooled as they are computed
instead of all 24 being held. 3D `E5L5S5` with `max` pooling peaks at 5.8 GB of MLX
memory instead of 20.4 GB.

**Separable wavelets.** Each 1-D pass convolves slabs of 4,194,304 output voxels
instead of the whole volume, passes are evaluated one at a time, and results shared
between bands are released after the last band that uses them. One `coif1` level-1
band peaks at 2,770 MB of MLX memory instead of 11,170 MB, and all eight bands at
3,497 MB instead of 28,475 MB (5,120 MB instead of 26,714 MB with 24-rotation
average pooling).

**Shape.** The mesh triangulates only cells on the boundary, and oriented bounding
box projections are evaluated in bounded groups. The geometric shape features of the
liver add 459 MB to the process footprint instead of 724 MB.

**Whole-image normalization holds one working copy.** Loading a 512×512×99 CT with
`normalize=True` peaked at 840 MiB and now peaks at 638 MiB on an M1 Max, with
identical values.

**Reading compressible NIfTI images no longer accumulates memory.** Each read of a
gzip-compressed float image with a zero background left about 20 MiB behind: eight
reads of a liver-only float32 CT raised the process peak from 309 to 456 MiB. The
validation pass now reads 1 MiB blocks and the peak stays at 264 MiB.

### Faster

**Diameters of large meshes.** A mesh of 32,768 or more vertices searches only the
vertices that can reach a maximum, and returns the same four diameters bit for bit.
The liver mesh (491,776 vertices) takes 168 ms instead of 367 ms on an M1 Max.

**Voxel-map planning scans the ROI bounding box instead of the whole grid.**
First-order maps of an 18,114-voxel tumor in the same CT take 196 ms instead of
311 ms.

## 1.1.0

### Breaking changes

**`mlxradiomics.medmlx.run` returns a `medmlx_core.outputs.RunResult`: `outputs` by role, `artifacts` and `details`.** `outputs` maps each role to a path: `features` (JSON report), `features_table` (CSV, ROI mode) and one role per map in voxel mode. The `files` key is gone, and `run_study` rows carry `outputs` and `artifacts` in its place (both `None` for a failed case). `run` is also importable from `mlxradiomics`.

```diff
-result["files"]["features_json"]
+result["outputs"]["features"]
-result["files"]["features_csv"]
+result["outputs"]["features_table"]
-result["files"]["maps"][name]
+result["outputs"][name]
```

**`temporary_dir` and `max_temporary_mib` are now `scratch_dir` and `max_scratch_mib`** in `run`, `export`, `export_files` and `run_study`.

```diff
-export_files(..., temporary_dir="/Volumes/fast", max_temporary_mib=4096)
+export_files(..., scratch_dir="/Volumes/fast", max_scratch_mib=4096)
```

**Requires `medmlx-core>=0.3,<0.4`** (1.0.1 required `>=0.2.6,<0.3`).

### New

**`extract` and `extract_files` accept `map_dtype="float32"` in voxel mode**, which retains maps at half the storage with the same values, special values and zero background. The default stays float64.

### Lower memory

**Reading NIfTI inputs peaks lower.** The finiteness check reads in 16 MiB blocks instead of loading the whole scaled volume; source validation peak RSS fell from 179 to 82 MB over three image and mask pairs on a DGX Spark.

**Selections that read no gray-level lattice (shape, raw first-order statistics, local intensity, IVH) no longer allocate an all-zero int32 volume.**

### Faster

**`run_study` reads each case's image and mask once and reuses them for every label of that case.**

## 1.0.1

The README and getting-started guide install with
`uv pip install "mlxradiomics>=1.0.1"`, matching medmlx.org. There are no code
changes; 1.0.0 users do not need to upgrade.

## 1.0.0

First stable release. mlxradiomics extracts radiomics features and voxel maps on
the Apple Silicon GPU with native MLX, using PyRadiomics feature names and
parameter files with IBSI conventions by default.

- **Four entry points.** `extract`, `extract_files`, `export` and `export_files`
  cover arrays and NIfTI/NRRD files, in memory and on disk, with one settings and
  result contract.
- **190 features.** All 116 non-deprecated PyRadiomics features plus IBSI
  intensity-histogram, intensity-volume histogram, local intensity, NGLDM and GLDZM
  families, in ROI and voxel mode.
- **Derived images.** Original, square, square root, logarithm, exponential,
  gradient, wavelet, LoG, LBP2D, LBP3D, mean, Laws, Gabor and separable wavelet
  images, each with typed per-filter options.
- **Configurable conventions.** Kurtosis, shape covariance divisor, NGTDM singular
  cases, fixed-bin-size origin and other definitions are individual `Settings`
  fields; see the [Python guide](guide/python-api.md#conventions).
- **Voxel maps.** Stream maps to a sink, or export float32 NIfTI maps with bounded
  host storage.
- **MedMLX runner.** `mlxradiomics.medmlx.run` reads files and writes a JSON/CSV
  report or maps; see the [runner guide](guide/runner.md).
- **Figures.** The optional `viz` extra exports feature-map figures as
  PNG, PDF or SVG.

The Simoncelli and Riesz filters are unavailable in this release; see
[Limits](guide/limits.md).
