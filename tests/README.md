# Real-image reference tests

Real-image tests use the converted HCC-TACE-Seg HCC_063 contrast CT, liver
mask, and tumor mask from
`~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`, `hcc_063_liver.nii.gz`, and
`hcc_063_tumor.nii.gz`. `MEDMLX_TEST_CT`, `MEDMLX_TEST_MASK`, and
`MEDMLX_TEST_TUMOR` override those paths.
The reference helper checks their shape, affine, and CT spacing before use.

File-I/O cases place source crops on controlled grids to exercise spacing,
orientation, encoding, and restoration. Geometry stress tests resize source crops
with nearest neighbors when their controlled grids exceed the source scan. Numerical stress cases apply declared
intensity scales or targeted corruptions to source pixels. Constant-field
cases use the scan's stored background. The IBSI operator checks express CT
in 512-HU units to retain their unit-scale error bounds.
Fourier checks select measured CT profile harmonics. Contingency tables,
rotation matrices, label selections, and precision operands remain explicit
mathematical controls.
