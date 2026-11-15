# Release notes

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
-export_files(..., voxel=True, temporary_dir="/Volumes/fast", max_temporary_mib=4096)
+export_files(..., voxel=True, scratch_dir="/Volumes/fast", max_scratch_mib=4096)
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
