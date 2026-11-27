# Limits

**Modes.** Shape and local-intensity features are ROI-only. Local intensity and
intensity-dependent morphology need 3D input. `shape2D` on 3D input needs
`force2d_axis` (a NumPy axis) and an ROI in one slice on that axis. Voxel mode
needs at least two ROI voxels.

**Filters.** The default recursive LoG needs 3D input with at least four samples
per axis and sigma/spacing at most 200. LBP3D needs 3D input and a masked kernel.
Integer inputs are limited to 8 or 16 bits. Samples near a rounding threshold
are evaluated in float64 with SciPy. An integer input whose sample remains
tied at the threshold raises `ValueError`. Floating-point inputs must pass the
source-precision checks for unambiguous binary comparisons.

**Numerics.** Feature arithmetic is float32 on Metal; binning is float64 on the
host. Undefined features are NaN, for example `shape_CentreOfMassShift` for
intensities that sum to exactly zero and the approximate-enclosing-ellipsoid
densities for a single-slice or collinear ROI. Float32 overflow gives infinity.
Inputs too ill-conditioned for float32 raise a precision error instead of returning
a wrong value.

**Files.** Read voxel maps with NiBabel: SimpleITK reads stored NaN voxels as zero.
NIfTI/Analyze header pairs named `.nii`/`.nii.gz` are rejected. When an input has
detached data (such as `.nhdr`), existing outputs are never replaced, even with
`overwrite=True`; use a new `output_dir`.
