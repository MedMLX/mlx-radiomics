"""IBSI 2 convolutional filters on Metal: mean, Laws, Gabor, LoG, wavelets, Simoncelli and Riesz.

The filter definitions follow the IBSI 2 reference manual (Depeursinge et al.,
"Standardised convolutional filtering for radiomics", arXiv:2006.05470v9,
sections 4.2, 4.3, 4.4, 4.5, 4.6.4 and 4.7). Filter kernels are prepared on the
host; every response-map convolution and frequency-domain operation runs in
float32 on MLX.

The FFT-based Simoncelli and Riesz response functions need MLX 0.32.5 or later:
earlier releases can overrun a batched FFT's destination buffer and corrupt
neighbouring arrays. On an older MLX the response functions raise ``ValueError``
(:data:`FFT_DISABLED_REASON`) before allocating a spectrum. ``FFT_AVAILABLE``
reports the gate. LoG, separable wavelets and LBP live in their own modules.

Two discrete Fourier grids are available. The DFT grid samples each axis with
the manual's exact ``2*pi/N`` step and places an exact zero-frequency bin at
index zero. The centered grid samples ``(k - (N - 1) / 2) / ((N - 1) / 2)``,
which is endpoint-inclusive and differs from the DFT grid by half a bin for even
lengths. The IBSI 2 Simoncelli and Riesz consensus maps were produced with the
centered grid, so it is exposed for reference comparisons; the DFT grid remains
the default because it is the grid specified by the manual's implementation
text and has an exact DC bin.
"""

from mlxradiomics.filters.convolution import gabor_kernel as gabor_kernel
from mlxradiomics.filters.convolution import gabor_responses as gabor_responses
from mlxradiomics.filters.convolution import laws_energy as laws_energy
from mlxradiomics.filters.convolution import laws_response as laws_response
from mlxradiomics.filters.convolution import mean_response as mean_response
from mlxradiomics.filters.convolution import pad_image as pad_image
from mlxradiomics.filters.convolution import (
    separable_wavelet_responses as separable_wavelet_responses,
)
from mlxradiomics.filters.spectral import FFT_AVAILABLE as FFT_AVAILABLE
from mlxradiomics.filters.spectral import FFT_DISABLED_REASON as FFT_DISABLED_REASON
from mlxradiomics.filters.spectral import log_response as log_response
from mlxradiomics.filters.spectral import require_isotropic_spacing as require_isotropic_spacing
from mlxradiomics.filters.spectral import riesz_log_responses as riesz_log_responses
from mlxradiomics.filters.spectral import riesz_multiplier as riesz_multiplier
from mlxradiomics.filters.spectral import riesz_simoncelli_responses as riesz_simoncelli_responses
from mlxradiomics.filters.spectral import simoncelli_response as simoncelli_response
