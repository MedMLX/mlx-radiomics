# Release notes

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
