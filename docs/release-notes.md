# Release notes

## Unreleased

### New

**Moment invariants.** The new `momentinvariant` family adds ten 3D ROI features: five moments of the ROI shape and five of its intensity that do not change under translation, rotation or uniform scaling, as defined in PySERA. `mlxradiomics.FEATURES` now lists 200 names.

## 1.2.0

### Fixes

**Incomplete NIfTI images and masks raise `ValueError`.** This covers files shorter than their header declares and compressed files with a damaged gzip stream.

**Invalid parameters fail before extraction.** Empty `featureClass` lists, repeated YAML keys, and unsupported NumPy scalar types now produce errors that name the offending field or key. IVH errors specify the integer intensity and range-bound requirements.

**Export and study errors preserve completed work.** A failed scratch write raises `OSError`. If a case exceeds the ROI working-memory budget, `run_study` records the failure and continues. Scratch errors name `scratch_dir` or `max_scratch_mib`.

**Undefined shape features return NaN.** This covers `shape_CentreOfMassShift` when the intensity sum is zero and approximate enclosing ellipsoid densities for rank-deficient ROIs.

**Texture arithmetic handles large gray-level counts.** NGTDM contrast avoids integer overflow at 46,342 or more occupied levels. Texture variances square by multiplication, and ROI GLCM MCC uses a new Metal eigensolver. The two arithmetic changes can affect the last digits of previously computed features.

**LBP3D resolves near-threshold samples in integer CT and MR ROIs.** Samples close to a rounding boundary are evaluated in float64 with SciPy. An exact tie still raises `ValueError`.

### Faster

Voxel texture maps use fused Metal kernels and share zone labels when their
domains match. ROI extraction batches texture directions and limits filter
output to the required window.

| Workload | 1.1.0 warm | 1.2.0 warm | Speedup |
| --- | ---: | ---: | ---: |
| HCC tumor, two unmasked voxel maps | 2,059.1 ms | 12.2 ms | 168.6× |
| HCC tumor crop, Laws energy and two ROI statistics | 600.7 ms | 26.0 ms | 23.1× |
| HCC liver crop, six radius-one texture maps | 334.1 ms | 48.1 ms | 6.94× |
| LUNG1 tumor, 107 ROI features | 130.4 ms | 19.5 ms | 6.67× |
| HCC tumor, 107 ROI features | 44.1 ms | 12.7 ms | 3.46× |
| HCC liver, 107 ROI features | 229.9 ms | 115.0 ms | 2.00× |
| HCC tumor crop, three radius-two texture maps | 29.0 ms | 25.2 ms | 1.15× |

Warm medians from six paired rounds on an Apple M5 Max.

### Lower memory

**Large derived images use smaller intermediate buffers.** Recursive LoG filters along native axes, Laws pools rotations as they are evaluated, and separable wavelets run in slabs. Three wavelet bands on a full HCC CT added 4.63 GiB to peak process memory, down from 21.11 GiB on an M5 Max.

[Benchmark inputs and measurements](https://github.com/MedMLX/mlx-radiomics/blob/v1.2.0/docs/technical.md#120-extraction-timings)

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
