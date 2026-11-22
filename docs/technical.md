# Technical notes

## Apple silicon extraction timings

The published `mlxradiomics` 1.1.0 wheel (`b58783c`) and the optimized build
(`8575200`) ran `extract_files` on the IBSI chest CT radiomics phantom and
its 125,256-voxel tumor mask. Both builds used Python 3.13.17, MLX 0.32.4,
medmlx-core 0.3.0, and SciPy 1.18.1 with TF32 disabled.

Times are warm wall-clock medians over six paired rounds on each Mac and
include file loading. Each trial ran in a fresh process, with three timed
calls for ROI workloads and one for voxel workloads after the first call.
Speedup is the before time divided by the after time.

| Workload | M1 Max before → after (ms) | Speedup | M5 Max before → after (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Default ROI, 93 features | 307.9 → 292.5 | 1.05× (inconclusive) | 131.8 → 125.1 | 1.05× |
| Filtered ROI, four image types | 586.2 → 442.1 | 1.33× | 227.2 → 199.8 | 1.14× |
| Four streamed voxel maps, radius 1 | 18,093.0 → 95.3 | 189.8× | 9,411.6 → 59.9 | 157.2× |
| 32 streamed run and zone maps, radius 2 | 23,209.3 → 837.4 | 27.7× | 11,534.5 → 199.3 | 57.9× |

The filtered ROI workload extracted mean, entropy, GLCM contrast, and GLCM MCC
from the original, gradient, mean-filtered, and LoG images (2 mm sigma). The
radius-1 workload streamed first-order mean and 10th percentile, GLCM contrast,
and GLRLM run entropy. Radius 2 streamed all GLRLM and GLSZM features.
Both voxel workloads used a 64 MiB scratch budget.

ROI values were identical on both Macs. Streamed voxel coordinates and values
passed parity checks at `atol=2e-7` and `rtol=2e-6`.
