# Technical notes

## 1.2.0 extraction timings

The published 1.1.0 wheel and the 1.2.0 runtime ran on an Apple M5 Max with
Python 3.13.17, MLX 0.32.4, and SciPy 1.16.3. The fixtures were HCC-TACE-Seg
HCC_063 contrast CT and its liver and tumor annotations, NSCLC-Radiomics
LUNG1-001 CT and its tumor mask, and MSD Task04 Hippocampus case 001, label 1.

The table gives warm medians from six paired rounds in fresh processes.
Array workloads include preprocessing and completed extraction. The NIfTI
rows also include file decoding. Peak memory is the additional physical
footprint above the process footprint after input setup.

| Workload | 1.1.0 (ms) | 1.2.0 (ms) | Speedup |
| --- | ---: | ---: | ---: |
| HCC tumor, two unmasked voxel maps | 2,059.1 | 12.2 | 168.6× |
| HCC tumor crop, Laws energy and two ROI statistics | 600.7 | 26.0 | 23.1× |
| HCC liver crop, six radius-one texture maps | 334.1 | 48.1 | 6.94× |
| LUNG1 tumor, 107 ROI features | 130.4 | 19.5 | 6.67× |
| Hippocampus, 107 ROI features | 33.5 | 6.0 | 5.61× |
| HCC tumor, 107 ROI features | 44.1 | 12.7 | 3.46× |
| HCC liver, 107 ROI features | 229.9 | 115.0 | 2.00× |
| HCC tumor crop, three radius-two zone and dependence maps | 29.0 | 25.2 | 1.15× |
| HCC tumor NIfTI, first-order mean | 204.3 | 158.1 | 1.29× |
| Full HCC CT, three separable wavelet bands | 580.6 | 455.3 | 1.28× |

The unmasked maps were GLCM contrast and first-order mean on a
192 × 256 × 48 crop containing the 18,114-voxel tumor. The wavelet case used
the Haar LLL, HLH, and HHH bands with first-order mean and variance.
Additional peak memory was 21.11 GiB in 1.1.0 and 4.63 GiB in 1.2.0.
The six liver texture maps added 94.9 MiB and 163.3 MiB, respectively.

Across 25 paired workloads, 16 matched exactly and nine matched within
`atol=2e-6` and `rtol=2e-5`. Warm latency improved in 23. Timing differences
for recursive LoG and normalization with resegmentation were inconclusive.
The 1.1.0 wheel rejected the tested integer LBP3D crop at a rounding threshold.
The 1.2.0 runtime completed it in 9.26 ms warm.

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
