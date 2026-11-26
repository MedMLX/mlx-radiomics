# Parameter files

The runner, studies and {func}`~mlxradiomics.config.parameters.parse_params`
accept PyRadiomics-style settings as a mapping or a YAML file. The supported
sections are `setting`, `imageType` and `featureClass`. Unsupported keys, image
types and per-image overrides are rejected, not ignored.

Mapping keys must be strings at every level. Numeric settings require numbers;
YAML `true` and `false` are accepted only for boolean settings.

```yaml
setting:
  binWidth: 25
  label: 1
  resampledPixelSpacing: [1.0, 1.0, 1.0]
  interpolator: sitkLinear
  padDistance: 1
imageType:
  Original: {}
  Wavelet:
    level: 1
featureClass:
  firstorder: [Mean, Entropy]
  glcm: [Contrast, JointEntropy]
```

`resampledPixelSpacing` uses ITK x-first order, as in PyRadiomics. The Python
`ImageSettings.resampled_spacing` uses NumPy axis order instead.

## Image types

`imageType` keys: `Original`, `Square`, `SquareRoot`, `Logarithm`, `Exponential`,
`Gradient`, `Wavelet`, `LoG`, `LBP2D`, `LBP3D`, `Mean`, `Laws`, `Gabor`,
`SeparableWavelet`. `Simoncelli` (alias `NonseparableWavelet`) and `Riesz` are
recognized but cannot run on this release; see [Limits](limits.md).

{func}`~mlxradiomics.config.parameters.parse_params` documents every per-filter key.
{mod}`mlxradiomics.config.transforms` documents the typed filter options. See
[Conventions](python-api.md#conventions) for the corresponding definition settings.

## From Python

```python
from mlxradiomics.config.parameters import load_params, parse_params

parsed = parse_params(load_params("params.yaml"), ndim=3)
parsed.settings, parsed.image_settings, parsed.transforms
```

The three parsed objects can be passed straight to
{func}`~mlxradiomics.extract_files` and {func}`~mlxradiomics.export_files`.

## Differences from PyRadiomics

Defaults follow IBSI; [Conventions](python-api.md#conventions) lists the settings
that select another definition. These differences from PyRadiomics 3.0.1 hold
under every setting:

| Area | mlxradiomics | PyRadiomics |
| --- | --- | --- |
| GLRLM on an ROI one voxel thick | All 13 directions (4 in 2D), as IBSI defines them | Drops directions along the thin axis, so every GLRLM feature differs when a run is longer than one voxel |
| `Wavelet` with `level: 2` or `start_level: 1` | Stationary transform with filters dilated per level, as `pywt.swtn(level=2)` | Applies the level-1 filters again without dilation; `wavelet2-*` values differ, level 1 agrees |
| `MeshVolume`, `SurfaceArea` and features derived from them | Closed marching-cubes mesh | Mesh left open at some voxel configurations; values differed by up to 0.2% on the ROIs tested, more on ROIs of a few voxels |
| GLCM `MCC` with one occupied gray level | NaN, like other undefined features | 1 |
| LBP3D `lbp-3D-m<k>` images | Norm of the degree k−1 spherical-harmonic band, each icosphere vertex weighted by its solid angle | Unweighted vertex sums scale every level by (N/4π)^1.5 for N vertices, 6.11 at the default subdivision. Polar and azimuth angles reach `scipy.special.sph_harm` swapped, so `m2` and higher are not spherical-harmonic bands. On HCC CT ROIs `m1` is 6.1 and `m2` about 13 times the mlxradiomics value; `lbp-3D-k` agrees |
| Resampled integer images | Resampled intensities stay floating point | Cast back to the input pixel type, which truncates an int16 CT |
| LoG image names | `sigma: [1]` names the image `log-sigma-1-0-mm-3D` | `log-sigma-1-mm-3D`; `sigma: [1.0]` gives the same name in both |
