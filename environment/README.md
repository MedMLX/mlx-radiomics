# Development environment

On macOS arm64, install Python 3.13, [uv](https://docs.astral.sh/uv/) and the
Xcode Command Line Tools, then run from the repository root:

```bash
make env
make qa
make test TESTS=tests/test_preprocessing.py
```

`make env` synchronizes `uv.lock` into `.venv`. QA and tests call the provisioned
Python directly; pass `PYTHON=/path/to/python` to use another existing environment.
`make qa` runs Ruff and Pyright over source, tests and examples. Runtime pins
live in `pyproject.toml`. Independent mathematical checks in the tests use NumPy,
SciPy, scikit-image, PyWavelets and SimpleITK.

The development group and the optional `lbp` extra pin different SciPy releases,
so provision the extra in its own environment with
`uv sync --locked --no-dev --extra lbp`.

## Source layout

The package root contains only `__init__.py` (the public API), `medmlx.py`
(the MedMLX runner) and `py.typed`. Implementations are grouped by the work
they perform:

| Path under `src/mlxradiomics/` | Responsibility |
| --- | --- |
| `config/` | Host-only feature, spatial and per-filter settings; enums, wavelet coefficients and parameter parsing |
| `contracts/` | Extraction/export results, host arrays, streaming protocols and JSON/report types |
| `imaging/` | Volume I/O, physical grids, host input preparation and derived-image planning |
| `workflows/` | Extraction orchestration, ROI and derived-image preparation, shared report writing, publication and studies |
| `features/` | Feature formulas, source admission, texture matrices and aggregation; `first_order.py` and `glcm.py` own their respective statistics |
| `features/geometry/` | Mesh construction, covariance, diameters, convex hulls, bounding boxes and ellipsoids |
| `filters/` | Spatial convolution, sampled LoG, guarded spectral filters, LBP and recursive Gaussian implementations |
| `numerics/` | Shared numerical primitives, explicit MLX execution and Metal-call protocols |
| `voxel/` | Batch scheduling/extraction, neighborhood gathering, fused GLCM/texture kernels and bounded map storage |
| `viz/` | CPU display composition and Matplotlib feature-map figure export |

Start at `mlxradiomics.extract` for array inputs or `mlxradiomics.extract_files`
for file inputs. Both are implemented in `workflows/extraction.py`, which calls
`workflows/engine.py` for original-image admission, then `workflows/roi.py` or
`voxel/extraction.py` for computation. `workflows/derived.py` owns derived-image
sampling and source-precision preparation. Both Python export and the MedMLX
adapter use `workflows/reporting.py` for map streaming and atomic report writing.
Follow the selected feature family from there. Shared numerical primitives live in
`numerics/`; family-specific kernels stay beside their formulas.
`contracts/results.py` defines the common extraction and map-export results.
The IBSI manifest lives in `features/`, and the mesh table is beside
`features/geometry/mesh.py`. The wavelet table lives in `imaging/` and is read by
the host-only `config/wavelets.py`. `contracts/metadata.py` keeps the serialized
report fields stable independently of the Python configuration structure.

All imports are absolute. Ruff enforces this with `TID252` and
`ban-relative-imports = "all"`.
