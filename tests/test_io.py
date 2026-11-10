"""Independent physical-grid expectations for image-file input."""

from collections.abc import Mapping
from itertools import product
from pathlib import Path
from typing import Literal, Protocol, cast

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
from numpy.typing import NDArray

from mlxradiomics.config.settings import Settings
from mlxradiomics.config.spatial import IbsiInterpolation, ImageSettings, Interpolator
from mlxradiomics.config.transforms import ImageType, TransformSettings
from mlxradiomics.imaging.io import read_image, read_mask
from mlxradiomics.imaging.spatial import load_image_pair
from mlxradiomics.voxel.storage import voxel_map_store
from mlxradiomics.workflows.extraction import extract_files


class _NiftiHeader(Protocol):
    def set_slope_inter(self, slope: float, inter: float) -> None: ...


class _WritableNifti(Protocol):
    @property
    def header(self) -> _NiftiHeader: ...


class _NiftiWriter(Protocol):
    def Nifti1Image(
        self, dataobj: NDArray[np.int16], affine: NDArray[np.float64],
        *, header: object | None = None,
    ) -> _WritableNifti: ...

    def Nifti1Header(self, *, endianness: str) -> object: ...

    def save(self, image: _WritableNifti, filename: str) -> None: ...


def _write_pair(root: Path, image: sitk.Image, mask: sitk.Image) -> tuple[Path, Path]:
    paths = root / "image.nrrd", root / "mask.nrrd"
    sitk.WriteImage(image, paths[0])
    sitk.WriteImage(mask, paths[1])
    return paths


def _fixture(*, reflected: bool = False) -> tuple[sitk.Image, sitk.Image]:
    values = np.random.default_rng(71).normal(-50, 300, (7, 8, 9))
    labels = np.zeros(values.shape, dtype=np.uint8)
    labels[2:6, 2:7, 1:6] = 1
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(labels)
    image.SetSpacing((0.7, 1.5, 2.2))
    image.SetOrigin((25.0, -11.0, 9.0))
    cosine, sine = np.cos(0.37), np.sin(0.37)
    direction = np.array(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))
    if reflected:
        direction[:, 0] *= -1
    image.SetDirection(direction.ravel().tolist())
    mask.CopyInformation(image)
    return image, mask


_SITK_INTERPOLATORS = {
    Interpolator.NEAREST: sitk.sitkNearestNeighbor,
    Interpolator.LINEAR: sitk.sitkLinear,
    Interpolator.BSPLINE: sitk.sitkBSpline,
}


def _resample_pair_on_grid(
    image: sitk.Image,
    mask: sitk.Image,
    *,
    size: tuple[int, ...],
    spacing: tuple[float, ...],
    origin_index: tuple[float, ...],
    interpolator: Interpolator,
) -> tuple[sitk.Image, sitk.Image]:
    """Use standard ITK interpolation on an explicitly specified fixture grid."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(size)
    resampler.SetOutputSpacing(spacing)
    resampler.SetOutputOrigin(mask.TransformContinuousIndexToPhysicalPoint(origin_index))
    resampler.SetOutputDirection(mask.GetDirection())
    resampler.SetOutputPixelType(sitk.sitkFloat64)
    resampler.SetInterpolator(_SITK_INTERPOLATORS[interpolator])
    expected_image = resampler.Execute(image)
    resampler.SetOutputPixelType(mask.GetPixelID())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    return expected_image, resampler.Execute(mask)


def _fixture_expected_pair(
    image: sitk.Image, mask: sitk.Image, spacing: tuple[float, ...],
    interpolator: Interpolator,
) -> tuple[sitk.Image, sitk.Image]:
    """Expected grids for the fixed ROI [z=2:6, y=2:7, x=1:6], with one pad voxel.

    The isotropic grid covers x centers 3/14 through 103/14, y centers 1/2
    through 43/6, and z centers 7/11 through 67/11 in source-index coordinates.
    The other cases retain source spacing and crop to x=0:7, y=1:8, z=1:7.
    """
    if spacing != (1.0, 1.0, 1.0):
        return (sitk.RegionOfInterest(image, [7, 7, 6], [0, 1, 1]),
                sitk.RegionOfInterest(mask, [7, 7, 6], [0, 1, 1]))
    # NRRD normalizes direction columns, which can move spacing by an ulp. The known
    # lower output-grid indices are (0, 1, 2); evaluate their half-voxel displacement
    # from the stored spacing, not simplified decimal fractions. A nearest-neighbor
    # sample exactly at a half index can otherwise select the opposite voxel.
    source = np.asarray(mask.GetSpacing())
    origin_index = 0.5 * (1 - source) / source + np.array([0, 1, 2]) / source
    return _resample_pair_on_grid(
        image, mask, size=(6, 11, 13), spacing=(1.0, 1.0, 1.0),
        origin_index=tuple(float(value) for value in origin_index), interpolator=interpolator,
    )


def _normalized_values(
    image: NDArray[np.float64],
    *,
    scale: float,
    outliers: float | None,
) -> NDArray[np.float64]:
    """Return the documented whole-image z-score, outlier clip and scale in float64."""
    normalized = (image - np.mean(image)) / np.std(image, ddof=1)
    if outliers is not None:
        normalized = np.clip(normalized, -outliers, outliers)
    return normalized * scale


def _numpy_image(values: NDArray[np.float64], template: sitk.Image) -> sitk.Image:
    """Wrap a NumPy array with the template's physical geometry."""
    image = sitk.GetImageFromArray(values)
    image.CopyInformation(template)
    return image


def _gradient_magnitude(
    values: NDArray[np.float64],
    spacing: tuple[float, ...],
) -> NDArray[np.float64]:
    """Return the central-difference gradient magnitude with replicated edge voxels."""
    squared = np.zeros_like(values)
    for axis, step in enumerate(spacing):
        indexes = np.arange(values.shape[axis], dtype=np.int64)
        left = np.take(values, np.maximum(indexes - 1, 0), axis=axis)
        right = np.take(values, np.minimum(indexes + 1, values.shape[axis] - 1), axis=axis)
        squared += ((right - left) / (2 * step)) ** 2
    return np.sqrt(squared)


def _masked_neighborhood_means(
    values: NDArray[np.float64],
    labels: NDArray[np.generic],
    label: int,
    radius: int = 1,
) -> NDArray[np.float64]:
    """Return each ROI voxel's mean over ROI voxels in its cubic neighborhood.

    Positions outside the array are excluded, as a masked kernel requires. Centers
    outside the ROI remain 0, matching voxel-map backgrounds.
    """
    roi = np.asarray(labels) == label
    output = np.zeros(values.shape, dtype=np.float64)
    offsets = np.asarray(
        list(product(range(-radius, radius + 1), repeat=values.ndim)),
        dtype=np.int64,
    )
    shape = np.asarray(values.shape, dtype=np.int64)
    for center in np.argwhere(roi):
        neighbors = center + offsets
        inside = np.all((neighbors >= 0) & (neighbors < shape), axis=1)
        selected = neighbors[inside]
        selected = selected[roi[tuple(selected.T)]]
        if len(selected):
            output[tuple(center)] = float(np.mean(values[tuple(selected.T)]))
    return output


@pytest.mark.parametrize("spacing", [(1.0, 1.0, 1.0), (0.0, 1.5, 0.0), (2.2, 1.5, 0.7)])
@pytest.mark.parametrize("interpolator", list(Interpolator))
@pytest.mark.parametrize("reflected", [False, True])
def test_resampling_uses_physical_roi_grid_and_pixels(
    tmp_path: Path,
    spacing: tuple[float, ...],
    interpolator: Interpolator,
    reflected: bool,
) -> None:
    image, mask = _fixture(reflected=reflected)
    options = ImageSettings(resampled_spacing=spacing, interpolator=interpolator, pad_distance=1)
    paths = _write_pair(tmp_path, image, mask)
    pair = load_image_pair(*paths, image_settings=options)
    # Read the serialized volumes so both sides decode the same stored grid; the
    # expected grid and pixels then come from ROI geometry and SimpleITK, respectively.
    image, mask = sitk.ReadImage(paths[0]), sitk.ReadImage(paths[1])
    expected_image, expected_mask = _fixture_expected_pair(image, mask, spacing, interpolator)
    assert pair.image.array.shape == expected_image.GetSize()[::-1]
    assert pair.mask.array.shape == expected_mask.GetSize()[::-1]
    assert pair.image.spacing == expected_image.GetSpacing()[::-1]
    np.testing.assert_allclose(pair.image.origin, expected_image.GetOrigin(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        pair.image.direction, expected_image.GetDirection(), rtol=0, atol=1e-12
    )
    assert pair.mask.spacing == pair.image.spacing
    assert pair.mask.origin == pair.image.origin
    assert pair.mask.direction == pair.image.direction
    if spacing == (1.0, 1.0, 1.0):
        # Geometry has passed the independent fixture-grid assertions above. Evaluate
        # pixels on that admitted grid's exact floating representation: a few origin
        # ulps can otherwise choose opposite sides of a nearest-neighbor half-index tie.
        expected_image.SetOrigin(pair.image.origin)
        expected_image.SetSpacing(pair.image.spacing[::-1])
        expected_image.SetDirection(pair.image.direction)
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(expected_image)
        resampler.SetInterpolator(_SITK_INTERPOLATORS[interpolator])
        resampler.SetOutputPixelType(sitk.sitkFloat64)
        expected_image = resampler.Execute(image)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetOutputPixelType(mask.GetPixelID())
        expected_mask = resampler.Execute(mask)
    np.testing.assert_array_equal(pair.mask.array, sitk.GetArrayFromImage(expected_mask))
    np.testing.assert_allclose(
        pair.image.array.astype(np.float64),
        sitk.GetArrayFromImage(expected_image).astype(np.float64),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("filtered", [False, True])
@pytest.mark.parametrize("voxel", [False, True], ids=["roi", "streamed"])
def test_normalization_precedes_resampling_and_runs_once(
    tmp_path: Path,
    filtered: bool,
    voxel: bool,
) -> None:
    image, mask = _fixture()
    paths = _write_pair(tmp_path, image, mask)
    options = Settings(
        normalize=True, normalize_scale=50, remove_outliers=2, features=("firstorder_Mean",)
    )
    spatial = ImageSettings(resampled_spacing=(1.0, 1.0, 1.0), pad_distance=1)
    transforms = TransformSettings(
        image_types=(ImageType.GRADIENT if filtered else ImageType.ORIGINAL,),
    )
    source_image, source_mask = sitk.ReadImage(paths[0]), sitk.ReadImage(paths[1])
    # Independent float64 normalization of the full source, then one padded resampling
    # onto the ROI-derived grid. A second normalization or resampling-before-normalization
    # would move both the mean and the gradient away from these values.
    values = _normalized_values(
        np.asarray(sitk.GetArrayFromImage(source_image), dtype=np.float64),
        scale=options.normalize_scale,
        outliers=options.remove_outliers,
    )
    expected, expected_mask = _fixture_expected_pair(
        _numpy_image(values, source_image), source_mask, (1.0, 1.0, 1.0),
        Interpolator.BSPLINE,
    )
    expected_values = np.asarray(sitk.GetArrayFromImage(expected), dtype=np.float64)
    labels = np.asarray(sitk.GetArrayFromImage(expected_mask))
    if filtered:
        expected_values = _gradient_magnitude(expected_values, expected.GetSpacing()[::-1])
    name = "gradient" if filtered else "original"
    key = f"{name}_firstorder_Mean"
    if voxel:
        expected_map = _masked_neighborhood_means(expected_values, labels, 1)
        with voxel_map_store(expected_values.shape, (key,)) as store:
            result = extract_files(
                *paths,
                settings=options,
                image_settings=spatial,
                transforms=transforms,
                mode="voxel",
                sink=store.append,
            )
            assert not result.features
            actual = {name: mapped.copy() for name, mapped in store.maps()}[key]
        np.testing.assert_allclose(actual, expected_map, rtol=1e-4, atol=1e-4)
    else:
        result = extract_files(
            *paths, settings=options, image_settings=spatial, transforms=transforms
        )
        np.testing.assert_allclose(
            result.features[key],
            float(np.mean(expected_values[labels == 1])),
            rtol=1e-4,
            atol=1e-4,
        )
    source_geometry = result.metadata.get("source_geometry")
    processed_geometry = result.metadata.get("image_geometry")
    assert source_geometry is not None and processed_geometry is not None
    assert source_geometry["shape"] == source_image.GetSize()[::-1]
    assert processed_geometry["shape"] == expected.GetSize()[::-1]
    assert source_geometry["origin"] == source_image.GetOrigin()
    assert processed_geometry["origin"] == expected.GetOrigin()
    spatial_metadata = result.metadata.get("image_settings")
    assert spatial_metadata is not None
    assert spatial_metadata["resampled_spacing"] == spatial.resampled_spacing


def _oblique_pair(ndim: int, label: int = 17) -> tuple[sitk.Image, sitk.Image]:
    """Create an asymmetric grid with an oblique reflection and known integer support."""
    shape = (7, 8, 9)[-ndim:]
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) - 50
    labels = np.zeros(shape, dtype=np.uint64)
    labels[tuple(slice(2, size - 2) for size in shape)] = label
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(labels)
    direction = np.eye(ndim)
    cosine, sine = np.cos(0.37), np.sin(0.37)
    direction[:2, :2] = ((-cosine, -sine), (-sine, cosine))
    image.SetDirection(direction.ravel().tolist())
    image.SetSpacing((0.7, 1.3, 2.2)[:ndim])
    image.SetOrigin((11.0, -7.0, 4.0)[:ndim])
    mask.CopyInformation(image)
    return image, mask


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("crop_only", [False, True])
def test_streamed_coordinates_use_the_oblique_processed_grid(
    tmp_path: Path,
    ndim: int,
    crop_only: bool,
) -> None:
    image, mask = _oblique_pair(ndim)
    paths = _write_pair(tmp_path, image, mask)
    image, mask = sitk.ReadImage(paths[0]), sitk.ReadImage(paths[1])
    spacing = image.GetSpacing()[::-1] if crop_only else (1.0,) * ndim
    spatial = ImageSettings(
        resampled_spacing=spacing, interpolator=Interpolator.LINEAR, pad_distance=1
    )
    options = Settings(label=17, features=("firstorder_Mean",))
    # Fixed support is x=2:7, y=2:6 (and z=2:5 in 3D). Cropping adds
    # one source voxel; isotropic resampling uses the explicit physical grid below.
    if crop_only:
        size = (7, 6, 5)[:ndim]
        expected = sitk.RegionOfInterest(image, list(size), [1] * ndim)
        expected_mask = sitk.RegionOfInterest(mask, list(size), [1] * ndim)
    else:
        expected, expected_mask = _resample_pair_on_grid(
            image, mask, size=(7, 10, 10)[:ndim], spacing=(1.0,) * ndim,
            origin_index=(3 / 14, -3 / 26, 7 / 11)[:ndim],
            interpolator=Interpolator.LINEAR,
        )
    values = np.asarray(sitk.GetArrayFromImage(expected), dtype=np.float64)
    labels = np.asarray(sitk.GetArrayFromImage(expected_mask))
    expected_values = _masked_neighborhood_means(values, labels, 17)
    collected = np.zeros(values.shape, dtype=np.float32)
    coordinates: list[NDArray[np.int32]] = []

    def consume(centers: NDArray[np.int32], batch: Mapping[str, NDArray[np.float32]]) -> None:
        coordinates.append(centers.copy())
        collected[tuple(centers.T)] = batch["original_firstorder_Mean"]

    result = extract_files(
        *paths, settings=options, image_settings=spatial, mode="voxel", sink=consume
    )
    np.testing.assert_allclose(collected, expected_values, rtol=1e-4, atol=1e-4)
    assert not result.features
    centers = np.concatenate(coordinates)
    selected = np.argwhere(labels == 17)
    assert len(centers) == len(selected)
    assert {tuple(center) for center in centers} == {tuple(center) for center in selected}
    assert "image_geometry" in result.metadata
    geometry = result.metadata["image_geometry"]
    assert geometry["shape"] == values.shape
    # The callback uses NumPy coordinates; physical positions use ITK x/y/z.
    physical = (
        np.asarray(geometry["origin"])
        + (centers[:, ::-1] * geometry["spacing"][::-1])
        @ np.asarray(geometry["direction"]).reshape(ndim, ndim).T
    )
    independent = np.array(
        [
            expected.TransformContinuousIndexToPhysicalPoint(center[::-1].astype(float).tolist())
            for center in centers
        ]
    )
    np.testing.assert_allclose(physical, independent, rtol=0, atol=1e-12)


@pytest.mark.parametrize("ndim", [2, 3])
def test_mask_correction_preserves_integer_labels_on_the_image_grid(
    tmp_path: Path, ndim: int,
) -> None:
    label = 2**32 - 1
    image, mask = _oblique_pair(ndim, label)
    labels = sitk.GetArrayFromImage(mask)
    direction = np.array(image.GetDirection()).reshape(ndim, ndim)
    mask.SetOrigin((np.array(image.GetOrigin()) + direction[:, 0] * image.GetSpacing()[0]).tolist())
    paths = _write_pair(tmp_path, image, mask)
    options = Settings(label=label)
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_image_pair(*paths, settings=options)
    pair = load_image_pair(*paths, settings=options,
                           image_settings=ImageSettings(correct_mask=True))
    expected = np.zeros(labels.shape, dtype=labels.dtype)
    expected[..., 1:] = labels[..., :-1]
    np.testing.assert_array_equal(pair.mask.array, expected)
    assert pair.mask.spacing == pair.image.spacing
    assert pair.mask.origin == pair.image.origin
    assert pair.mask.direction == pair.image.direction
    mask.SetOrigin((np.array(image.GetOrigin()) + 1000).tolist())
    with pytest.raises(ValueError, match="outside the physical image bounds"):
        load_image_pair(*_write_pair(tmp_path, image, mask), settings=options,
                        image_settings=ImageSettings(correct_mask=True))


def test_geometry_match_is_judged_in_voxels_at_micrometre_spacing(tmp_path: Path) -> None:
    # 0.1 um voxels in metres: a mask origin 5 voxels along x (5e-7) is a different grid
    # even though every coordinate differs by less than 1e-6.
    spacing = 1e-7
    image = sitk.GetImageFromArray(np.zeros((8, 8, 8), dtype=np.float32))
    image.SetSpacing((spacing,) * 3)
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2, 2, 2] = 1
    mask = sitk.GetImageFromArray(labels)
    mask.SetSpacing((spacing,) * 3)
    mask.SetOrigin((5 * spacing, 0.0, 0.0))
    paths = _write_pair(tmp_path, image, mask)
    with pytest.raises(ValueError, match="geometry mismatch"):
        load_image_pair(*paths)
    pair = load_image_pair(*paths, image_settings=ImageSettings(correct_mask=True))
    # Mask voxel (z, y, x) = (2, 2, 2) lies at physical x = 5e-7 + 2e-7 = image x index 7.
    expected = np.zeros_like(labels)
    expected[2, 2, 7] = 1
    np.testing.assert_array_equal(pair.mask.array, expected)


@pytest.mark.parametrize("mode", ["crop", "resample", "correct"])
def test_spatial_paths_do_not_merge_large_adjacent_labels(tmp_path: Path, mode: str) -> None:
    label = 2**53
    image, mask = _oblique_pair(2, label)
    labels = sitk.GetArrayFromImage(mask)
    labels[2:4, 2:4] = label + 1
    mask = sitk.GetImageFromArray(labels)
    mask.CopyInformation(image)
    options = Settings(label=label)
    if mode != "crop":
        if mode == "correct":
            direction = np.array(image.GetDirection()).reshape(2, 2)
            mask.SetOrigin((np.array(image.GetOrigin()) +
                            direction[:, 0] * image.GetSpacing()[0]).tolist())
            spatial = ImageSettings(correct_mask=True)
        else:
            spatial = ImageSettings(resampled_spacing=(0.9, 0.9), pad_distance=0)
        with pytest.raises(ValueError, match="exact integer.*53"):
            load_image_pair(*_write_pair(tmp_path, image, mask), settings=options,
                            image_settings=spatial)
    else:
        # Same-grid cropping needs no interpolator and must keep the full integer dtype.
        spatial = ImageSettings(resampled_spacing=image.GetSpacing()[::-1], pad_distance=20)
        pair = load_image_pair(*_write_pair(tmp_path, image, mask), settings=options,
                               image_settings=spatial)
        np.testing.assert_array_equal(pair.mask.array, labels)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("endian", ["<", ">"])
def test_scaled_nifti_applies_scaling_and_lps_axis_conventions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ndim: int, endian: str,
) -> None:
    raw = np.arange(np.prod((4, 5, 6)[:ndim]), dtype=np.int16).reshape((4, 5, 6)[:ndim])
    affine = np.eye(4)
    cosine, sine = np.cos(0.37), np.sin(0.37)
    direction = np.array(((-cosine, -sine, 0), (-sine, cosine, 0), (0, 0, 1)))
    affine[:3, :3] = direction @ np.diag((0.7, 1.3, 2.2))
    affine[:3, 3] = (11.0, -7.0, 4.0)
    writer = cast(_NiftiWriter, nib)
    node = writer.Nifti1Image(raw, affine, header=writer.Nifti1Header(endianness=endian))
    node.header.set_slope_inter(2.5, -1024)
    path = tmp_path / "scaled.nii.gz"
    writer.save(node, str(path))
    monkeypatch.setattr("mlxradiomics.imaging.io._NIFTI_VALIDATION_BYTES", 16)
    volume = read_image(path)
    np.testing.assert_array_equal(volume.array, raw.T * 2.5 - 1024)
    lps = np.diag((-1, -1, 1))[:ndim, :ndim]
    expected_origin = lps @ affine[:ndim, 3]
    expected_direction = lps @ direction[:ndim, :ndim]
    np.testing.assert_allclose(volume.spacing, (0.7, 1.3, 2.2)[:ndim][::-1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(volume.origin, expected_origin, rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.array(volume.direction).reshape(ndim, ndim), expected_direction,
                               rtol=0, atol=1e-6)
    with pytest.raises(ValueError, match="integer labels"):
        read_mask(path)


@pytest.mark.parametrize("suffix", [".nrrd", ".nii.gz"])
@pytest.mark.parametrize("label", [2**32 - 1, 2**53 + 1, 2**64 - 1])
def test_file_masks_preserve_uint64_label_values(tmp_path: Path, suffix: str, label: int) -> None:
    labels = np.zeros((5, 6), dtype=np.uint64)
    labels[1:4, 1:5] = label
    path = tmp_path / f"mask{suffix}"
    sitk.WriteImage(sitk.GetImageFromArray(labels), path)
    np.testing.assert_array_equal(read_mask(path).array, labels)


@pytest.mark.parametrize("suffix", [".nrrd", ".nii", ".nii.gz"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_file_reader_rejects_nonfinite_source_voxels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str, nonfinite: float,
) -> None:
    values = np.arange(30, dtype=np.float64).reshape(5, 6)
    values[-1, -1] = nonfinite
    path = tmp_path / f"nonfinite{suffix}"
    sitk.WriteImage(sitk.GetImageFromArray(values), path)
    monkeypatch.setattr("mlxradiomics.imaging.io._NIFTI_VALIDATION_BYTES", 16)
    with pytest.raises(ValueError, match="non-finite"):
        read_image(path)


@pytest.mark.parametrize("kind", ["floating_mask", "vector_image", "four_dimensions"])
def test_file_readers_reject_unsupported_pixel_contracts(tmp_path: Path, kind: str) -> None:
    if kind == "floating_mask":
        image = sitk.GetImageFromArray(np.ones((5, 6), dtype=np.float64))
        reader, message = read_mask, "integer labels"
    else:
        image = sitk.GetImageFromArray(np.zeros((2, 3, 4, 5), dtype=np.float32),
                                       isVector=kind == "vector_image")
        reader = read_image
        message = "single-channel" if kind == "vector_image" else "2D or 3D"
    path = tmp_path / "unsupported.nrrd"
    sitk.WriteImage(image, path)
    with pytest.raises(ValueError, match=message):
        reader(path)


def _ibsi_linear_pair() -> tuple[sitk.Image, sitk.Image]:
    """Create an oblique linear-ramp image and a binary mask inside it."""
    shape, spacing, origin = (5, 6, 7), (0.75, 1.4, 2.5), (10.0, -5.0, 3.0)
    cosine, sine = np.cos(0.37), np.sin(0.37)
    direction = np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
    index = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), axis=-1)
    physical = origin + (index * spacing) @ direction.T
    values = 3 * physical[..., 0] + 5 * physical[..., 1] - 2 * physical[..., 2] + 7
    labels = np.zeros(shape, dtype=np.uint8)
    labels[1:4, 2:5, 2:6] = 1
    image = sitk.GetImageFromArray(values.transpose(2, 1, 0))
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    image.SetDirection(direction.ravel().tolist())
    mask = sitk.GetImageFromArray(labels.transpose(2, 1, 0))
    mask.CopyInformation(image)
    return image, mask


def _ibsi_nonlinear_pair() -> tuple[sitk.Image, sitk.Image]:
    """Create an oblique nonlinear image and a non-1-label mask for cubic interpolation."""
    shape = (8, 9)
    row, column = np.indices(shape, dtype=np.float64)
    values = 3.0 * column**2 + 2.0 * row**3 + 5.0 * row * column - 30.0 + 0.5
    labels = np.zeros(shape, dtype=np.uint8)
    labels[1:7, 1:8] = 5
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(labels)
    image.SetSpacing((1.0, 1.0))
    image.SetOrigin((3.0, -2.0))
    cosine, sine = np.cos(0.37), np.sin(0.37)
    image.SetDirection((cosine, -sine, sine, cosine))
    mask.CopyInformation(image)
    return image, mask


def _ibsi_bspline_reference(
    image: sitk.Image, shape: tuple[int, ...], spacing: tuple[float, ...],
    origin: tuple[float, ...],
) -> NDArray[np.float64]:
    """Resample with standard SimpleITK cubic B-spline behavior for an independent check."""
    reference = sitk.ResampleImageFilter()
    reference.SetSize(list(shape))
    reference.SetOutputSpacing(list(spacing))
    reference.SetOutputOrigin(list(origin))
    reference.SetOutputDirection(image.GetDirection())
    reference.SetOutputPixelType(sitk.sitkFloat64)
    reference.SetInterpolator(sitk.sitkBSpline)
    reference.SetUseNearestNeighborExtrapolator(True)
    return np.asarray(
        sitk.GetArrayFromImage(reference.Execute(sitk.Cast(image, sitk.sitkFloat64))),
        dtype=np.float64,
    )


def test_ibsi_centered_grid_matches_closed_form(tmp_path: Path) -> None:
    image, mask = _ibsi_linear_pair()
    # Target spacing keeps every output center inside the source center extent, so the
    # linear ramp is reproduced exactly without boundary clamping.
    options = ImageSettings(
        ibsi_interpolation=IbsiInterpolation(spacing=(3.0, 1.75, 1.0), round_intensities=False),
    )
    pair = load_image_pair(*_write_pair(tmp_path, image, mask), image_settings=options)
    source_spacing, target = np.asarray(image.GetSpacing()), np.array((1.0, 1.75, 3.0))
    size = np.asarray(image.GetSize())
    # The fixed fixture has target extents 3.75, 4.8, and 35/6 cells in ITK order.
    shape = (4, 5, 6)
    direction = np.asarray(image.GetDirection()).reshape(3, 3)
    expected_origin = np.asarray(image.GetOrigin()) + direction @ (
        (source_spacing * (size - 1) - target * (np.asarray(shape) - 1)) / 2
    )
    assert pair.image.array.shape == shape[::-1]
    np.testing.assert_allclose(pair.image.spacing, target[::-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(pair.image.origin, expected_origin, rtol=0, atol=1e-12)
    np.testing.assert_allclose(pair.image.direction, image.GetDirection(), rtol=0, atol=1e-12)
    index = np.moveaxis(np.indices(shape, dtype=np.float64), 0, -1)
    physical = expected_origin + (index * target) @ direction.T
    expected = 3 * physical[..., 0] + 5 * physical[..., 1] - 2 * physical[..., 2] + 7
    np.testing.assert_allclose(pair.image.array.astype(np.float64), expected.transpose(2, 1, 0),
                               rtol=0, atol=1e-9)


def test_ibsi_bspline_image_keeps_linear_mask_and_rounding(tmp_path: Path) -> None:
    image, mask = _ibsi_nonlinear_pair()
    paths = _write_pair(tmp_path, image, mask)
    target = (2.0, 2.0)
    settings = Settings(label=5, spline_boundary="mirror")
    linear = load_image_pair(*paths, settings=settings, image_settings=ImageSettings(
        ibsi_interpolation=IbsiInterpolation(spacing=target),
    ))
    cubic_unrounded = load_image_pair(*paths, settings=settings, image_settings=ImageSettings(
        ibsi_interpolation=IbsiInterpolation(
            spacing=target, image_interpolator=Interpolator.BSPLINE,
            round_intensities=False,
        ),
    ))
    cubic = load_image_pair(*paths, settings=settings, image_settings=ImageSettings(
        ibsi_interpolation=IbsiInterpolation(
            spacing=target, image_interpolator=Interpolator.BSPLINE,
        ),
    ))

    source = sitk.ReadImage(str(paths[0]))
    # The 9-by-8 unit grid becomes 5-by-4 cells at spacing two, with a half-voxel
    # displacement along its second axis to keep the physical center fixed.
    shape, spacing = (5, 4), (2.0, 2.0)
    direction = np.asarray(source.GetDirection()).reshape(2, 2)
    shifted_origin = np.asarray(source.GetOrigin()) + direction @ np.array((0., 0.5))
    origin = (float(shifted_origin[0]), float(shifted_origin[1]))
    assert cubic.image.array.shape == shape[::-1]
    np.testing.assert_allclose(cubic.image.spacing, spacing[::-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cubic.image.origin, origin, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cubic.image.direction, source.GetDirection(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(cubic.mask.spacing, spacing[::-1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(cubic.mask.origin, origin, rtol=0, atol=1e-12)
    np.testing.assert_allclose(cubic.mask.direction, source.GetDirection(), rtol=0, atol=1e-12)

    # The mask is always the linear occupancy path, even for a non-1 label; a cubic
    # choice leaking into the reused resampler would change its thresholded voxels.
    np.testing.assert_array_equal(cubic.mask.array, linear.mask.array)
    assert cubic.mask.array.dtype == np.uint8
    assert set(np.unique(cubic.mask.array)) == {0, 5}
    assert np.count_nonzero(cubic.mask.array) > 0

    # Cubic interpolation must differ from the linear default on this nonlinear image.
    assert np.any(cubic.image.array != linear.image.array)

    # Compare the unrounded public path with standard SimpleITK direct behavior, then
    # verify the public path rounds the source and result to nearest integers, ties even.
    expected_unrounded = _ibsi_bspline_reference(source, shape, spacing, origin)
    np.testing.assert_allclose(cubic_unrounded.image.array.astype(np.float64),
                               expected_unrounded, rtol=0, atol=1e-12)
    rounded_source = sitk.GetImageFromArray(np.rint(sitk.GetArrayFromImage(source)))
    rounded_source.CopyInformation(source)
    expected_rounded = _ibsi_bspline_reference(rounded_source, shape, spacing, origin)
    np.testing.assert_array_equal(cubic.image.array, np.rint(expected_rounded))


@pytest.mark.parametrize("label", [1, 5])
def test_ibsi_interpolation_pads_edges_and_thresholds_only_the_selected_label(
    tmp_path: Path, label: int,
) -> None:
    values = np.tile((np.arange(4) * 10.0)[None, :], (4, 1))
    labels = np.zeros((4, 4), dtype=np.uint8)
    labels[1:3, 1:3] = label
    # An adjacent region must not blend its label magnitude into selected occupancy.
    labels[0, 0] = 9
    image, mask = sitk.GetImageFromArray(values), sitk.GetImageFromArray(labels)
    image.SetSpacing((3.0, 3.0))
    mask.CopyInformation(image)
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(2.0, 2.0)))
    pair = load_image_pair(*_write_pair(tmp_path, image, mask), settings=Settings(label=label),
                           image_settings=options)
    # Output centers are at input indexes -1/6, 1/2, 7/6, 11/6, 5/2, 19/6. The ramp
    # gives 0, 5, 35/3, 55/3, 25 and a boundary-clamped 30; rounding then applies.
    expected = np.tile(np.array((0.0, 5.0, 12.0, 18.0, 25.0, 30.0)), (6, 1))
    np.testing.assert_array_equal(pair.image.array, expected)
    assert pair.mask.array.dtype == np.uint8
    assert set(np.unique(pair.mask.array)) == {0, label}
    # Axis occupancies are 1/2, 1, 1 and 1/2. The four outer corners have product
    # 1/4, below threshold; the remaining 12 voxels meet the 1/2 threshold.
    expected_mask = np.zeros((6, 6), dtype=np.uint8)
    expected_mask[1:5, 2:4] = label
    expected_mask[2:4, 1:5] = label
    np.testing.assert_array_equal(pair.mask.array, expected_mask)
    assert pair.mask.spacing == (2.0, 2.0)


@pytest.mark.parametrize("threshold", [0.7, np.float64(0.7)])
def test_ibsi_mask_threshold_includes_exactly_covered_voxels(
    tmp_path: Path, threshold: float,
) -> None:
    labels = np.zeros((4, 10), dtype=np.uint8)
    labels[:, :2] = 1
    image, mask = sitk.GetImageFromArray(np.zeros((4, 10))), sitk.GetImageFromArray(labels)
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(
        spacing=(0.0, 0.4), mask_threshold=threshold, round_intensities=False,
    ))
    pair = load_image_pair(
        *_write_pair(tmp_path, image, mask),
        settings=Settings(mask_extrapolation="nearest"), image_settings=options,
    )
    # 25 centers at input columns -0.3 + 0.4 k. Occupancy is 1 through column 1, then
    # 1 - (x - 1): column 1.3 is covered exactly 0.7 (float32 0.699999988) and meets a
    # 0.7 threshold; column 1.7 (0.3) does not. NumPy scalar thresholds must not change it.
    expected = np.zeros((4, 25), dtype=np.uint8)
    expected[:, :5] = 1
    np.testing.assert_array_equal(pair.mask.array, expected)


def test_ibsi_mask_extrapolation_zero_excludes_centers_beyond_source_centers(
    tmp_path: Path,
) -> None:
    # Output centers at input columns -0.3 + 0.4 k: the first (-0.3) lies beyond the
    # outermost source voxel center, as do 9.9-ish ones at the far end of a full mask.
    labels = np.zeros((4, 10), dtype=np.uint8)
    labels[:, :2] = 1
    labels[:, 8:] = 1
    image, mask = sitk.GetImageFromArray(np.zeros((4, 10))), sitk.GetImageFromArray(labels)
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(
        spacing=(0.0, 0.4), round_intensities=False,
    ))
    paths = _write_pair(tmp_path, image, mask)
    zero = load_image_pair(*paths, settings=Settings(), image_settings=options)
    nearest = load_image_pair(
        *paths, settings=Settings(mask_extrapolation="nearest"), image_settings=options,
    )
    centers = -0.3 + 0.4 * np.arange(25)
    inside = (centers >= 0) & (centers <= 9)
    # The two settings differ only at output centers outside [0, 9] in input columns.
    assert np.array_equal(zero.mask.array[:, inside], nearest.mask.array[:, inside])
    assert not zero.mask.array[:, ~inside].any()
    assert nearest.mask.array[:, ~inside].any()
    # Image intensities are extrapolated with the nearest source value either way.
    np.testing.assert_array_equal(zero.image.array, nearest.image.array)


def test_ibsi_interpolation_preserves_large_binary_labels(tmp_path: Path) -> None:
    label = 2**32 - 1
    image, mask = _oblique_pair(3, label)
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(1.0, 1.0, 1.0)))
    pair = load_image_pair(*_write_pair(tmp_path, image, mask), settings=Settings(label=label),
                           image_settings=options)
    assert pair.mask.array.dtype == np.uint64
    assert set(np.unique(pair.mask.array)) == {0, label}
    assert np.count_nonzero(pair.mask.array == label) > 0


def test_ibsi_interpolation_rejects_unsafe_integer_labels(tmp_path: Path) -> None:
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(1.0, 1.0, 1.0)))
    image, mask = _oblique_pair(3, 2**53 + 1)
    with pytest.raises(ValueError, match="exact integer.*53"):
        load_image_pair(*_write_pair(tmp_path, image, mask), settings=Settings(label=2**53 + 1),
                        image_settings=options)


def test_ibsi_rounding_refuses_normalized_intensities(tmp_path: Path) -> None:
    # Rounding z-scores to integers would silently collapse every intensity feature.
    image, mask = _oblique_pair(3)
    paths = _write_pair(tmp_path, image, mask)
    rounded = ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="round_intensities=False"):
        load_image_pair(*paths, settings=Settings(label=17, normalize=True),
                        image_settings=rounded)
    exact = ImageSettings(ibsi_interpolation=IbsiInterpolation(
        spacing=(1.0, 1.0, 1.0), round_intensities=False,
    ))
    pair = load_image_pair(*paths, settings=Settings(label=17, normalize=True),
                           image_settings=exact)
    values = np.asarray(pair.image.array, dtype=np.float64)
    assert not np.all(values == np.round(values))


def test_ibsi_settings_reject_conflicting_or_invalid_options() -> None:
    interpolation = IbsiInterpolation(spacing=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="mutually exclusive"):
        ImageSettings(resampled_spacing=(1.0, 1.0, 1.0), ibsi_interpolation=interpolation)
    with pytest.raises(ValueError, match="correct_mask"):
        ImageSettings(correct_mask=True, ibsi_interpolation=interpolation)
    with pytest.raises(ValueError, match="mask_threshold"):
        IbsiInterpolation(spacing=(1.0, 1.0), mask_threshold=0.0)
    with pytest.raises(ValueError, match="image_interpolator"):
        IbsiInterpolation(spacing=(1.0, 1.0), image_interpolator=Interpolator.NEAREST)
    with pytest.raises(ValueError, match="image_interpolator"):
        IbsiInterpolation(
            spacing=(1.0, 1.0), image_interpolator=cast(Interpolator, "linear"),
        )


@pytest.mark.parametrize("explicit", [False, True])
def test_ibsi_zero_spacing_keeps_the_source_grid(tmp_path: Path, explicit: bool) -> None:
    image, mask = _oblique_pair(2)
    paths = _write_pair(tmp_path, image, mask)
    options = ImageSettings(
        ibsi_interpolation=IbsiInterpolation(
            spacing=image.GetSpacing()[::-1] if explicit else (0.0, 0.0),
            round_intensities=False,
        ),
    )
    pair = load_image_pair(*paths, settings=Settings(label=17), image_settings=options)
    source, expected_mask = sitk.ReadImage(paths[0]), sitk.ReadImage(paths[1])
    np.testing.assert_allclose(
        pair.image.array.astype(np.float64),
        sitk.GetArrayFromImage(source).astype(np.float64),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_array_equal(pair.mask.array, sitk.GetArrayFromImage(expected_mask))
    assert pair.image.array.shape == sitk.GetArrayFromImage(source).shape
    assert pair.image.spacing == source.GetSpacing()[::-1]
    assert pair.image.origin == source.GetOrigin()
    np.testing.assert_allclose(pair.image.direction, source.GetDirection(), rtol=0, atol=1e-15)


@pytest.mark.parametrize("size, spacing, target, expected", [
    (3, 0.1, 0.1, 3), (245, 0.977, 0.977, 245),
    (6, 0.1, 0.2, 3), (3, 0.1, 0.05, 6), (490, 0.977, 1.954, 245),
    (5, 0.8, 1.0, 5),  # Stored binary64 0.8 is greater than 4/5.
    (6, 0.1, float(np.nextafter(0.2, 0)), 4),
    (6, 0.1, float(np.nextafter(0.2, np.inf)), 3),
])
def test_ibsi_exact_spacing_ratio_preserves_dimensions(
    tmp_path: Path, size: int, spacing: float, target: float, expected: int,
) -> None:
    image = sitk.GetImageFromArray(np.ones((size, size)))
    image.SetSpacing((spacing, spacing))
    mask = sitk.GetImageFromArray(np.ones((size, size), np.uint8))
    mask.CopyInformation(image)
    pair = load_image_pair(*_write_pair(tmp_path, image, mask), image_settings=ImageSettings(
        ibsi_interpolation=IbsiInterpolation(spacing=(target, target))))
    assert pair.image.array.shape == (expected, expected)
    origin = (spacing * (size - 1) - target * (expected - 1)) / 2
    np.testing.assert_allclose(pair.image.origin, (origin, origin), rtol=0, atol=1e-14)


@pytest.mark.parametrize("spacing, size", [(1.9999999999, 3), (4e10, 1)])
def test_ibsi_ceiling_retains_fractional_extents_and_small_images(
    tmp_path: Path, spacing: float, size: int,
) -> None:
    image = sitk.GetImageFromArray(np.ones((4, 4), dtype=np.float64))
    mask = sitk.GetImageFromArray(np.ones((4, 4), dtype=np.uint8))
    options = ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(spacing, spacing)))
    pair = load_image_pair(
        *_write_pair(tmp_path, image, mask),
        settings=Settings(mask_extrapolation="nearest"), image_settings=options,
    )
    # A true ratio just above 2 must ceil to 3; a positive ratio below 1 still needs one
    # voxel. Both cases would be reduced incorrectly by subtracting a fixed tolerance.
    assert pair.image.array.shape == (size, size)
    np.testing.assert_array_equal(pair.image.array, np.ones((size, size)))
    np.testing.assert_array_equal(pair.mask.array, np.ones((size, size), dtype=np.uint8))


def test_ordinary_resampling_is_independent_of_integer_file_storage(tmp_path: Path) -> None:
    # A four-value ramp resampled from unit to half spacing samples source indexes
    # -1/4, 1/4, ..., 13/4. Linear interpolation (with the source value clamped outside)
    # gives [0, 1/4, 3/4, 5/4, 7/4, 9/4, 11/4, 3] and mean 3/2. An integer-valued output
    # buffer would truncate those quarters and change the radiomic input to mean 9/8.
    ramp = np.broadcast_to(np.arange(4, dtype=np.float32), (4, 4)).copy()
    mask = np.ones((4, 4), dtype=np.uint8)
    mask_path = tmp_path / "mask.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(mask), mask_path)
    expected_row = np.array((0.0, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.0))
    options = ImageSettings(
        resampled_spacing=(0.5, 0.5), interpolator=Interpolator.LINEAR, pad_distance=0,
    )
    arrays: list[NDArray[np.float64]] = []
    means: list[float] = []
    for dtype in (np.int16, np.float32):
        image_path = tmp_path / f"ramp_{np.dtype(dtype).name}.nrrd"
        sitk.WriteImage(sitk.GetImageFromArray(ramp.astype(dtype)), image_path)
        pair = load_image_pair(image_path, mask_path, image_settings=options)
        values = np.asarray(pair.image.array, dtype=np.float64)
        arrays.append(values)
        assert pair.image.array.dtype == np.dtype(np.float64)
        assert pair.mask.array.dtype == np.dtype(np.uint8)
        assert pair.image.array.shape == (8, 8)
        np.testing.assert_allclose(
            values, np.broadcast_to(expected_row, values.shape), rtol=0, atol=1e-12,
        )
        result = extract_files(
            image_path, mask_path, settings=Settings(features=("firstorder_Mean",)),
            image_settings=options,
        )
        means.append(float(result.features["original_firstorder_Mean"]))
    np.testing.assert_allclose(arrays[0], arrays[1], rtol=0, atol=1e-12)
    np.testing.assert_allclose(means, (1.5, 1.5), rtol=0, atol=1e-6)


@pytest.mark.parametrize("ibsi", [False, True])
def test_spatial_loading_refuses_lossy_64_bit_integers_before_resampling(
    tmp_path: Path, ibsi: bool,
) -> None:
    # Resampling casts to float64 (and IBSI rounding follows) before extraction's own
    # integer check could see that 2**53 + 1 had already become 2**53.
    image = np.array([[[2**53, 2**53 + 1], [2**53, 2**53 + 1]]] * 2, np.int64)
    paths = (tmp_path / "image.nrrd", tmp_path / "mask.nrrd")
    sitk.WriteImage(sitk.GetImageFromArray(image), str(paths[0]))
    sitk.WriteImage(sitk.GetImageFromArray(np.ones(image.shape, np.uint8)), str(paths[1]))
    options = (ImageSettings(ibsi_interpolation=IbsiInterpolation(spacing=(1.0, 1.0, 1.0)))
               if ibsi else ImageSettings(resampled_spacing=(1.0, 1.0, 1.0)))
    with pytest.raises(ValueError, match="cannot represent exactly"):
        load_image_pair(*paths, image_settings=options)


def test_ibsi_bspline_nearest_boundary_matches_scipy_map_coordinates(tmp_path: Path) -> None:
    from scipy.ndimage import map_coordinates

    values = np.random.default_rng(0).normal(0, 100, (9, 11, 13))
    image = sitk.GetImageFromArray(values)
    image.SetSpacing((1.3, 1.7, 3.0))
    mask = sitk.GetImageFromArray(np.ones(values.shape, np.uint8))
    mask.CopyInformation(image)
    paths = _write_pair(tmp_path, image, mask)
    target = (1.0, 1.0, 1.0)

    def load(
        boundary: Literal["mirror", "nearest"], interpolator: Interpolator,
    ) -> NDArray[np.float64]:
        pair = load_image_pair(
            *paths,
            settings=Settings(spline_boundary=boundary),
            image_settings=ImageSettings(ibsi_interpolation=IbsiInterpolation(
                spacing=target, round_intensities=False,
                image_interpolator=interpolator,
            )),
        )
        return np.asarray(pair.image.array, dtype=np.float64)

    nearest = load("nearest", Interpolator.BSPLINE)
    # Source continuous index of each output center, from the loaded output geometry.
    source = sitk.ReadImage(str(paths[0]))
    pair = load_image_pair(*paths, image_settings=ImageSettings(
        ibsi_interpolation=IbsiInterpolation(spacing=target, round_intensities=False,
                                             image_interpolator=Interpolator.BSPLINE)))
    axes: list[NDArray[np.float64]] = [
        (pair.image.origin[k] + np.arange(nearest.shape[::-1][k], dtype=np.float64)
         * pair.image.spacing[k] - source.GetOrigin()[k]) / source.GetSpacing()[k]
        for k in range(3)
    ]
    grids: list[NDArray[np.float64]] = list(np.meshgrid(*axes[::-1], indexing="ij"))
    expected = np.asarray(map_coordinates(
        values, np.stack([g.ravel() for g in grids]), order=3, mode="nearest",
    ), dtype=np.float64).reshape(nearest.shape)
    np.testing.assert_allclose(nearest, expected, rtol=0, atol=1e-8)
    assert np.abs(load("mirror", Interpolator.BSPLINE) - expected).max() > 1.0
    np.testing.assert_array_equal(
        load("nearest", Interpolator.LINEAR), load("mirror", Interpolator.LINEAR),
    )
