"""HCC-TACE-Seg CT and source-aligned anatomical masks for reference tests."""

from __future__ import annotations

import os
from collections.abc import Callable
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
from scipy.ndimage import zoom


class ImageProxy(Protocol):
    def __getitem__(self, key: tuple[slice | int, ...]) -> npt.ArrayLike: ...


@lru_cache(maxsize=1)
def _images(ct_path: str, liver_path: str, tumor_path: str) -> tuple[
    nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image,
]:
    """Load the CT and its masks after checking their source-grid geometry."""
    load = cast(Callable[[str], object], import_module("nibabel").load)
    images = tuple(load(path) for path in (ct_path, liver_path, tumor_path))
    if not all(isinstance(image, nib.Nifti1Image) for image in images):
        raise ValueError("HCC-TACE inputs must be NIfTI images")
    ct, liver, tumor = cast(tuple[nib.Nifti1Image, nib.Nifti1Image, nib.Nifti1Image], images)
    if ct.affine is None or liver.affine is None or tumor.affine is None:
        raise ValueError("HCC-TACE images must have spatial affines")
    if ct.shape != liver.shape or ct.shape != tumor.shape:
        raise ValueError("HCC-TACE CT and mask shapes do not match")
    if (not np.allclose(ct.affine, liver.affine, rtol=0, atol=1e-5)
            or not np.allclose(ct.affine, tumor.affine, rtol=0, atol=1e-5)):
        raise ValueError("HCC-TACE CT and mask affines do not match")
    if not np.allclose(ct.header.get_zooms()[:3], (0.859375, 0.859375, 2.5), rtol=0, atol=1e-6):
        raise ValueError("Expected the HCC_063 CT spacing in millimeters")
    return ct, liver, tumor


def hcc_crop(
    crop: tuple[slice | int, ...],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Return matching CT, liver, and tumor voxels in NIfTI array order."""
    cache = Path.home() / ".cache" / "medmlx" / "test-data"
    ct_path = os.environ.get("MEDMLX_TEST_CT", str(cache / "hcc_063_ct.nii.gz"))
    liver_path = os.environ.get("MEDMLX_TEST_MASK", str(cache / "hcc_063_liver.nii.gz"))
    tumor_path = os.environ.get("MEDMLX_TEST_TUMOR", str(cache / "hcc_063_tumor.nii.gz"))
    ct, liver, tumor = _images(ct_path, liver_path, tumor_path)
    values = np.asarray(cast(ImageProxy, ct.dataobj)[crop], dtype=np.float64)
    liver_mask = np.asarray(cast(ImageProxy, liver.dataobj)[crop], dtype=np.uint8)
    tumor_mask = np.asarray(cast(ImageProxy, tumor.dataobj)[crop], dtype=np.uint8)
    return values, liver_mask, tumor_mask


def hcc_patch(shape: tuple[int, ...]) -> tuple[
    npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.uint8],
]:
    """Read a 2D or 3D CT patch and its annotations without reshaping source voxels.

    Two-dimensional patches use slice 64. Three-dimensional patches start at
    voxel (103, 214, 60); axes retain the NIfTI array order of ``hcc_crop``.
    """
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        raise ValueError("A source CT patch needs two or three positive dimensions")
    selection: tuple[slice | int, ...] = tuple(
        slice(start, start + size) for start, size in zip((103, 214, 60), shape, strict=False)
    )
    if len(shape) == 2:
        selection += (64,)
    result = hcc_crop(selection)
    if result[0].shape != shape:
        raise ValueError("The source CT does not cover the requested patch")
    return result


def hcc_background(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Read a constant stored-background crop for singular-value checks."""
    if len(shape) not in (2, 3) or any(size <= 0 for size in shape):
        raise ValueError("A source background crop needs two or three positive dimensions")
    selection: tuple[slice | int, ...] = tuple(slice(0, size) for size in shape)
    if len(shape) == 2:
        selection += (64,)
    image, _, _ = hcc_crop(selection)
    if image.shape != shape or not np.all(image == -2048):
        raise ValueError("The requested crop is not constant HCC_063 stored background")
    return image


def hcc_resampled(shape: tuple[int, ...]) -> npt.NDArray[np.float64]:
    """Resize a source CT crop with nearest neighbors for controlled test grids.

    Geometry stress tests include axes longer than the source scan. Their grid
    and labels are controlled independently; every intensity remains a sampled
    source voxel. Ordinary image tests should use ``hcc_patch`` directly.
    """
    source = hcc_patch(tuple(min(size, 16) for size in shape))[0]
    factors = tuple(size / old for size, old in zip(shape, source.shape, strict=True))
    return np.asarray(zoom(source, factors, order=0, prefilter=False), dtype=np.float64)
