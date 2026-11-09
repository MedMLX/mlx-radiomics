"""Host-side voxel batch storage preserves output values and physical placement."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from numpy.typing import NDArray

import mlxradiomics
from mlxradiomics.imaging.io import Volume, read_image, write_feature_map
from mlxradiomics.voxel.storage import voxel_map_store


class _NiftiImage(Protocol):
    def get_fdata(self) -> NDArray[np.float64]: ...


class _NiftiReader(Protocol):
    def load(self, filename: str) -> _NiftiImage: ...


@pytest.mark.parametrize("ndim", [2, 3])
def test_spooled_maps_preserve_values_across_batches(tmp_path: Path, ndim: int) -> None:
    # Cross the writer and reader chunk boundaries and leave background voxels on the grid.
    shape = (257, 257) if ndim == 2 else (41, 41, 41)
    count = 65539
    centers = np.column_stack(np.unravel_index(np.arange(1, count + 1), shape)).astype(np.int32)
    # Exercise noncontiguous coordinate columns and repeated centers within and across blocks.
    centers = centers[:, ::-1]
    centers[11] = centers[10]
    centers[65536] = centers[10]
    centers[-1] = centers[10]
    names = ("original_firstorder_Mean", "original_glcm_Contrast")
    features = {
        names[0]: np.linspace(-100, 100, count, dtype=np.float32),
        names[1]: np.linspace(100, 0, count, dtype=np.float32),
    }
    features[names[0]][:4] = [np.nan, np.inf, -np.inf, -0.0]
    reference = Volume(
        array=np.zeros(shape, dtype=np.float32),
        spacing=(0.8, 1.3) if ndim == 2 else (2.2, 0.8, 1.3),
        origin=(3.0, -2.0) if ndim == 2 else (3.0, -2.0, 5.0),
        direction=(-0.8, -0.6, -0.6, 0.8) if ndim == 2 else
        (-0.8, -0.6, 0.0, -0.6, 0.8, 0.0, 0.0, 0.0, 1.0),
    )
    with voxel_map_store(shape, names) as store:
        for selection in (slice(0, 65537), slice(65537, None)):
            store.append(centers[selection], {n: v[selection] for n, v in features.items()})
        for name, values in store.maps():
            expected = np.zeros(shape, dtype=np.float32)
            expected[tuple(centers.T)] = features[name]
            np.testing.assert_array_equal(values, expected)
            assert values[tuple(centers[10])] == features[name][-1]
            path = tmp_path / f"{name}.nii.gz"
            write_feature_map(path, values, reference=reference)
            restored = cast(_NiftiReader, nib).load(str(path)).get_fdata().T
            np.testing.assert_array_equal(restored, expected)
            if name == names[0]:
                assert np.signbit(restored[tuple(centers[3])])
                # Borrowers may write the map; the next feature still needs zero background.
                values[(0,) * ndim] = 123
            else:
                volume = read_image(path)
                assert volume.array.shape == shape
                np.testing.assert_allclose(volume.spacing, reference.spacing, rtol=0, atol=1e-6)
                np.testing.assert_allclose(volume.origin, reference.origin, rtol=0, atol=1e-6)
                np.testing.assert_allclose(volume.direction, reference.direction, rtol=0, atol=1e-6)


@pytest.mark.parametrize(("axis", "coordinate"), [(1, -1), (2, 41)])
def test_invalid_late_coordinate_writes_no_partial_records(
    tmp_path: Path, axis: int, coordinate: int,
) -> None:
    # The invalid row follows a full valid block, so writing blocks before validating
    # would leak rows that the next map iteration reads as complete records.
    shape = (41, 41, 41)
    count = 65537
    centers = np.column_stack(np.unravel_index(np.arange(count), shape)).astype(np.int32)
    centers[-1, axis] = coordinate
    with voxel_map_store(shape, ("mean",), temporary_dir=tmp_path) as store:
        with pytest.raises(ValueError):
            store.append(centers, {"mean": np.arange(count, dtype=np.float32)})
        store.append(np.array([[2, 3, 4]], dtype=np.int32),
                     {"mean": np.array([7], dtype=np.float32)})
        _, output = next(store.maps())
        expected = np.zeros(shape, dtype=np.float32)
        expected[2, 3, 4] = 7
        np.testing.assert_array_equal(output, expected)


def test_feature_count_does_not_exhaust_file_descriptors(tmp_path: Path) -> None:
    """Exercise a real low descriptor limit in an isolated host-only process."""
    script = """
import resource
import numpy as np
from mlxradiomics.voxel.storage import voxel_map_store
_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (32, hard))
names = tuple(f'feature_{i}' for i in range(129))
centers = np.array([[0, 0], [1, 1]], dtype=np.int32)
features = {name: np.full(2, i, dtype=np.float32) for i, name in enumerate(names)}
with voxel_map_store((3, 3), names) as store:
    store.append(centers, features)
    for i, (name, values) in enumerate(store.maps()):
        assert name == names[i]
        assert values[0, 0] == i and values[1, 1] == i and values[2, 2] == 0
"""
    environment = os.environ.copy()
    # Keep child processes on the parent's source or installed-wheel package.
    environment["PYTHONPATH"] = str(Path(mlxradiomics.__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, "-c", script], env=environment,
                            capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("interruption", [ValueError, KeyboardInterrupt])
def test_scratch_cleanup_after_export_interruption(
    tmp_path: Path, interruption: type[BaseException],
) -> None:
    marker = tmp_path / "unrelated"
    marker.write_bytes(b"preserved")
    with pytest.raises(interruption), voxel_map_store((4, 5), ("mean",), temporary_dir=tmp_path,
                         max_temporary_bytes=1024) as store:
        store.append(np.array([[0, 0], [1, 1]], dtype=np.int32),
                     {"mean": np.array([2, np.nan], dtype=np.float32)})
        iterator = store.maps()
        _, values = next(iterator)
        assert values[0, 0] == 2 and np.isnan(values[1, 1])
        raise interruption()
    assert marker.read_bytes() == b"preserved"
    assert list(tmp_path.iterdir()) == [marker]


def test_scratch_budget_rejects_a_batch_before_partial_append(tmp_path: Path) -> None:
    # 80-byte dense image + 8-byte block header + four (8-byte flat index, 4-byte value) rows.
    with voxel_map_store((4, 5), ("mean",), temporary_dir=tmp_path,
                         max_temporary_bytes=135) as store:
        with pytest.raises(ValueError, match="temporary.*budget"):
            store.append(np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=np.int32),
                         {"mean": np.arange(4, dtype=np.float32)})
        # Rejection leaves the store usable for a smaller batch, with no rejected values.
        store.append(np.array([[1, 1]], dtype=np.int32), {"mean": np.array([7], dtype=np.float32)})
        _, output = next(store.maps())
        expected = np.zeros((4, 5), dtype=np.float32)
        expected[1, 1] = 7
        np.testing.assert_array_equal(output, expected)
    assert not list(tmp_path.iterdir())
