# Third-party notices

PyRadiomics-derived marching-geometry lookup tables in
`src/mlxradiomics/features/geometry/mesh_tables.json` (recorded source revision
`6a761c4eb01905cfcb1139b0b158547c55edd4df`, `radiomics/src/cshape.c`).
Source: https://raw.githubusercontent.com/AIM-Harvard/pyradiomics/6a761c4eb01905cfcb1139b0b158547c55edd4df/LICENSE.txt

Copyright 2017 Harvard Medical School

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The 3-D marching table rows 128-255 are derived from scikit-image 0.26.0's
`measure.marching_cubes` classic (`lorensen`) binary-cell geometry, with edge
indices remapped and winding reversed to point outward. Rows 0-127 and the
remaining geometry tables retain the PyRadiomics provenance above.
Source: https://github.com/scikit-image/scikit-image/tree/v0.26.0/skimage/measure
License: https://github.com/scikit-image/scikit-image/blob/v0.26.0/LICENSE.txt

Copyright 2009-2022 the scikit-image team

License: BSD-3-Clause

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of the University nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.
.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE HOLDERS OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Wavelet coefficient tables in `src/mlxradiomics/imaging/wavelet_tables.json` record
PyWavelets 1.10.0 as their source:
Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
Copyright (c) 2012-     The PyWavelets Developers <https://github.com/PyWavelets/pywt>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Documentation feature-map figure

The CT image and tumour contour in `docs/feature-maps.png` are case LUNG1-023 of
[NSCLC-Radiomics](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI), Aerts, Hugo J. W. L.
et al. (2014), The Cancer Imaging Archive, licensed under
[CC BY-NC 3.0](https://creativecommons.org/licenses/by-nc/3.0/). The figure shows a
cropped slice with feature maps computed by this package and is shared under the
same license. The image is not part of the distributed packages.

## IBSI feature definitions

The definition titles and permanent feature identifiers in
`docs/ibsi-features.json` are adapted from the Image Biomarker
Standardisation Initiative reference manual, by the IBSI contributors, at
revision `b952e71e9d40569e8a9ff3c740bc389c04dd6c28`:

- [IBSI feature definitions](https://github.com/theibsi/ibsi-doc/blob/b952e71e9d40569e8a9ff3c740bc389c04dd6c28/docs/03_Image_features.rst)
- [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

Adaptations add public feature-name mappings and use DICOM CID 7478 for
discretized percentile IDs `GPMT` and `OZ0C`. The CC BY 4.0 attribution applies to the adapted
manual material; existing software and table licenses remain as stated above.
