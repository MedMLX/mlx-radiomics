"""NIfTI/NRRD volume I/O with explicit value and geometry contracts.

Reading and writing use SimpleITK. NiBabel also checks NIfTI image values before
extraction because SimpleITK replaces stored non-finite voxels with zero.
"""

from __future__ import annotations

import dataclasses
import gzip
from collections.abc import Mapping
from math import prod
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from mlxradiomics.contracts.types import HostArray

_VOLUME_SUFFIXES = (".nii", ".nii.gz", ".nrrd", ".nhdr")
_NIFTI_VALIDATION_BYTES = 16 * 1024 * 1024


class _NiftiProxy(Protocol):
    """Public NiBabel proxy metadata and scaled slicing used for validation."""

    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]
    offset: int
    slope: float
    inter: float

    def __getitem__(self, key: slice, /) -> HostArray:
        """Read one scaled contiguous storage-order block."""
        ...


class _NiftiProxyFactory(Protocol):
    def __call__(
        self, file_like: object,
        spec: tuple[tuple[int, ...], np.dtype[np.generic], int, float, float],
        *, mmap: bool,
    ) -> _NiftiProxy:
        """Construct the standard proxy over an explicitly owned file stream."""
        ...


class _NiftiImage(Protocol):
    """The part of a NiBabel image used to inspect stored voxel values."""

    @property
    def dataobj(self) -> _NiftiProxy:
        """Expose scaled proxy data using its native dtype and required scaling precision."""
        ...


class _NiftiReader(Protocol):
    """The part of the ``nibabel`` module used to load NIfTI images."""

    def load(self, filename: str) -> _NiftiImage:
        """Load an image with its on-disk voxel values intact."""
        ...


@dataclasses.dataclass(frozen=True)
class Volume:
    """A 2D/3D volume in this package's array-axis convention (NumPy order).

    ``array`` is indexed in NumPy order and ``spacing`` (mm) has one entry per array axis,
    i.e. it is the reverse of ITK's x-first spacing. ``origin`` (mm) and ``direction`` keep
    the ITK physical x/y/z order; ``direction`` is the flattened row-major direction
    matrix and must be orthonormal.

    Raises:
        ValueError: On construction, if the array is empty, not 2-D or 3-D, or not real
            numeric, or if spacing, origin or direction are non-finite, have the wrong
            length, or (for spacing) are not positive.
    """

    array: HostArray
    spacing: tuple[float, ...]
    origin: tuple[float, ...]
    direction: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject arrays, spacing, origin or direction outside the documented contract."""
        ndim = self.array.ndim
        if ndim not in (2, 3) or any(size == 0 for size in self.array.shape):
            raise ValueError("volume must have a nonempty 2D or 3D shape")
        if self.array.dtype.kind not in "biuf":
            raise ValueError("volume must contain real numeric values")
        if len(self.spacing) != ndim or any(not np.isfinite(v) or v <= 0 for v in self.spacing):
            raise ValueError("volume requires finite positive spacing per axis")
        if len(self.origin) != ndim or not np.all(np.isfinite(self.origin)):
            raise ValueError("volume requires a finite origin per physical axis")
        if len(self.direction) != ndim**2 or not np.all(np.isfinite(self.direction)):
            raise ValueError("volume requires a finite square direction matrix")
        matrix = np.asarray(self.direction).reshape(ndim, ndim)
        if not np.allclose(matrix.T @ matrix, np.eye(ndim), rtol=0, atol=1e-5):
            raise ValueError("volume direction must be orthonormal")


def _require_volume_path(path: str | Path) -> Path:
    """Return ``path`` if it names an existing NIfTI or NRRD file, else raise ``ValueError``."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"missing input volume: {source}")
    if not source.name.endswith(_VOLUME_SUFFIXES):
        raise ValueError(f"expected a {'/'.join(_VOLUME_SUFFIXES)} file: {source}")
    return source


def _require_single_file_nifti(source: Path) -> None:
    """Reject paired NIfTI/Analyze headers disguised as single-file ``.nii`` inputs.

    SimpleITK reads header bytes as voxels for these names, so they must
    be converted to a single-file image or a supported NRRD before extraction.
    """
    with gzip.open(source, "rb") if source.name.endswith(".gz") else source.open("rb") as stream:
        header = stream.read(348)
    magic1, magic2 = header[344:348], header[4:8]
    header1 = header[:4] in (b"\x5c\x01\0\0", b"\0\0\x01\x5c")
    header2 = header[:4] in (b"\x1c\x02\0\0", b"\0\0\x02\x1c")
    analyze = header1 and magic1 not in (b"n+1\0", b"ni1\0")
    if (header1 and magic1 == b"ni1\0") or (header2 and magic2 == b"ni2\0") or analyze:
        raise ValueError(f"detached NIfTI/Analyze input is unsupported: {source}")


def _read(path: str | Path, *, integer: bool) -> Volume:
    """Read a volume with SimpleITK under mask (``integer``) or image rules.

    Masks must hold integer labels. Images must be finite; NIfTI images are also
    checked through NiBabel because SimpleITK zeroes stored non-finite voxels.
    """
    import SimpleITK as sitk

    source = _require_volume_path(path)
    if source.name.endswith((".nii", ".nii.gz")):
        _require_single_file_nifti(source)
    image = sitk.ReadImage(source.as_posix())
    if image.GetDimension() not in (2, 3):
        raise ValueError(f"expected a 2D or 3D volume: {source}")
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError(f"expected a single-channel volume: {source}")
    raw = sitk.GetArrayFromImage(image)
    spacing, origin, direction = tuple(reversed(image.GetSpacing())), image.GetOrigin(), (
        image.GetDirection()
    )
    # GetArrayFromImage owns its pixels; keep geometry, then release ITK's second
    # full buffer before checking the independent NIfTI value representation.
    del image
    if integer and raw.dtype.kind not in "biu":
        raise ValueError(f"mask must contain integer labels: {source}")
    array = np.asarray(raw)
    if not integer:
        flat = array.reshape(-1)
        count = max(1, _NIFTI_VALIDATION_BYTES // max(array.itemsize, 8))
        for start in range(0, array.size, count):
            if not np.all(np.isfinite(flat[start:start + count])):
                raise ValueError(f"image contains non-finite voxels: {source}")
    if not integer and source.name.endswith((".nii", ".nii.gz")):
        _validate_nifti_values(source)
    return Volume(
        array=array,
        # SimpleITK orders spacing/origin/direction x-first; this package's arrays
        # follow NumPy axis order, so spacing (but not origin/direction) is reversed.
        spacing=spacing,
        origin=origin,
        direction=direction,
    )


def _validate_nifti_values(source: Path) -> None:
    """Reject stored nonfinite values using bounded, forward-only proxy reads.

    SimpleITK zeroes stored NaN/infinity. Preserve NiBabel's native dtype and
    scaling, but validate a flat storage-order proxy over one owned stream so
    gzip decompression never restarts between blocks. NiBabel 5.4's ``reshape``
    loses ``keep_file_open``, so construct its documented proxy from public
    metadata instead. No pixel values or geometry are taken from this check.
    """
    import nibabel as nib
    from nibabel.arrayproxy import ArrayProxy

    proxy = cast(_NiftiReader, nib).load(source.as_posix()).dataobj
    size = prod(proxy.shape)
    # Scaling can promote to float64 (complex128 for a complex source, which the
    # volume contract later rejects). Bound both stored and scaled block payload.
    count = max(1, _NIFTI_VALIDATION_BYTES // np.result_type(proxy.dtype, np.float64).itemsize)
    factory = cast(_NiftiProxyFactory, ArrayProxy)
    with gzip.open(source, "rb") if source.name.endswith(".gz") else source.open("rb") as stream:
        flat = factory(
            stream, ((size,), proxy.dtype, proxy.offset, proxy.slope, proxy.inter), mmap=False,
        )
        for start in range(0, size, count):
            if not np.all(np.isfinite(flat[start:start + count])):
                raise ValueError(f"image contains non-finite voxels: {source}")


def read_image_dimension(path: str | Path) -> int:
    """Read image dimensionality from the header without loading or validating pixels.

    The runner needs this before parsing dimension-dependent parameters. Full image
    validation still belongs to ``read_image`` when spatial preprocessing loads pixels.
    """
    import SimpleITK as sitk

    source = _require_volume_path(path)
    if source.name.endswith((".nii", ".nii.gz")):
        _require_single_file_nifti(source)
    reader = sitk.ImageFileReader()
    reader.SetFileName(source.as_posix())
    reader.ReadImageInformation()
    return reader.GetDimension()


def read_image(path: str | Path) -> Volume:
    """Read a 2D/3D NIfTI or NRRD image preserving pixel type and physical geometry.

    SimpleITK and NiBabel read errors propagate for unreadable files.

    Raises:
        ValueError: If the file is missing, has an unsupported suffix or detached NIfTI
            header, is not a single-channel 2-D or 3-D volume, or contains non-finite values.
    """

    return _read(path, integer=False)


def read_mask(path: str | Path) -> Volume:
    """Read a 2D/3D integer label mask preserving pixel type and physical geometry.

    SimpleITK read errors propagate, including ``RuntimeError`` for unreadable files.

    Raises:
        ValueError: If the file is missing, has an unsupported suffix or detached NIfTI
            header, is not a single-channel 2-D or 3-D volume, or has a non-integer pixel type.
    """

    return _read(path, integer=True)


def has_detached_data(source: Path) -> bool:
    """Detect detached NRRD and NIfTI data by header, including compressed headers.

    Raises:
        ValueError: If the file header cannot be inspected or recognized.
    """
    try:
        with (
            gzip.open(source, "rb") if source.name.endswith(".gz") else source.open("rb")
        ) as header:
            prefix = header.read(348)
            if prefix.startswith(b"NRRD"):
                header.seek(0)
                for line in header:
                    if not line.strip():
                        break
                    field, separator, value = line.partition(b":")
                    if (separator and not value.startswith(b"=")
                            and field.strip().lower() in (b"data file", b"datafile")):
                        return True
                return False
            if prefix[344:348] in (b"n+1\0", b"ni1\0"):
                return prefix[344:348] == b"ni1\0"
            if prefix[4:8] in (b"n+2\0", b"ni2\0"):
                return prefix[4:8] == b"ni2\0"
            # SimpleITK also accepts Analyze 7.5 pairs renamed with a .nii suffix.
            if prefix[:4] in (b"\x5c\x01\0\0", b"\0\0\x01\x5c"):
                return True
    except (OSError, EOFError) as error:
        raise ValueError(f"cannot inspect volume header: {source}") from error
    raise ValueError(f"cannot inspect volume header: {source}")


class _NiftiMapHeader(Protocol):
    """The part of a NiBabel NIfTI-1 header written for feature maps."""

    def set_xyzt_units(self, xyz: str) -> None:
        """Set the spatial units code."""
        ...
    def set_slope_inter(self, slope: float, inter: float) -> None:
        """Set the stored value scaling."""
        ...


class _NiftiMap(Protocol):
    """The part of a NiBabel NIfTI-1 image written for feature maps."""

    @property
    def header(self) -> _NiftiMapHeader:
        """Return the image header."""
        ...

    def set_qform(self, affine: NDArray[np.float64], code: int) -> None:
        """Set the qform affine and code."""
        ...
    def set_sform(self, affine: NDArray[np.float64], code: int) -> None:
        """Set the sform affine and code."""
        ...
    def to_file_map(self, file_map: Mapping[str, object]) -> None:
        """Write the image to the given file holders."""
        ...


class _NiftiWriter(Protocol):
    """The part of the ``nibabel`` module used to write feature maps."""

    def Nifti1Image(
        self, dataobj: NDArray[np.float32] | NDArray[np.float64], affine: NDArray[np.float64],
        *, dtype: np.dtype[np.float32],
    ) -> _NiftiMap:
        """Create a NIfTI-1 image from data and an affine."""
        ...

    def FileHolder(self, *, fileobj: gzip.GzipFile) -> object:
        """Wrap an open file object for ``to_file_map``."""
        ...

    def save(self, image: _NiftiMap, filename: str) -> None:
        """Save ``image`` to ``filename``."""
        ...


def write_feature_map(
    path: str | Path,
    array: NDArray[np.float32] | NDArray[np.float64],
    *,
    reference: Volume,
) -> None:
    """Write one voxel-mode feature map as float32, reusing the reference volume's geometry.

    NIfTI uses NiBabel's slice writer over an array view, including disk-backed maps;
    it avoids SimpleITK's full-volume copy. ``.nii.gz`` streams through ISA-L deflate at
    level 1 instead of NiBabel's zlib stream; the
    gzip header keeps NiBabel's deterministic form (mtime 0, no stored filename), and
    the decompressed NIfTI bytes are unchanged. Other suffixes retain the SimpleITK writer.
    Missing parent directories are created. The SimpleITK NIfTI reader maps stored
    NaN and infinities to zero on read-back, even though those values are present in
    the written file; use a reader that preserves them when they matter.

    Raises:
        ValueError: If ``array`` does not have the shape of ``reference.array``.
    """

    if array.shape != reference.array.shape:
        raise ValueError("feature map shape must match the reference volume")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        ndim = array.ndim
        affine = np.eye(4)
        signs = np.array((-1.0, -1.0, 1.0))[:ndim]
        direction = np.asarray(reference.direction).reshape(ndim, ndim)
        # Reverse array axes for NIfTI storage, and convert ITK LPS physical axes to RAS.
        affine[:ndim, :ndim] = signs[:, None] * direction * np.asarray(reference.spacing[::-1])
        affine[:ndim, 3] = signs * reference.origin
        writer = cast(_NiftiWriter, nib)
        image = writer.Nifti1Image(array.T, affine, dtype=np.dtype(np.float32))
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=1)
        image.header.set_xyzt_units("mm")
        image.header.set_slope_inter(1.0, 0.0)
        if not destination.name.endswith(".gz"):
            writer.save(image, str(destination))
            return
        from isal.igzip import IGzipFile

        with (
            destination.open("wb") as raw,
            IGzipFile(filename="", mode="wb", compresslevel=1, fileobj=raw, mtime=0) as stream,
        ):
            image.to_file_map({"image": writer.FileHolder(fileobj=stream)})
        return

    import SimpleITK as sitk

    image = sitk.GetImageFromArray(np.asarray(array, dtype=np.float32))
    image.SetSpacing(tuple(reversed(reference.spacing)))
    image.SetOrigin(reference.origin)
    image.SetDirection(reference.direction)
    sitk.WriteImage(image, destination.as_posix())
