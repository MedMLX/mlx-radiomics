"""Bounded-chunk host storage for evaluated voxel batches and one disk-backed dense map."""

import errno
import mmap
import os
import shutil
from collections.abc import Generator, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from tempfile import TemporaryFile, gettempdir
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray


@dataclass
class VoxelMapStore:
    """Spool all selected features into one unlinked file, independent of feature count.

    Use ``voxel_map_store`` to own the files and mapping. Each compact block contains
    an int64 row count, int64 row-major flat voxel indices, then one float32 column per
    feature. Blocks hold at most 65,536 rows. This is process-local scratch, not an
    interchange format.
    """

    shape: tuple[int, ...]
    names: tuple[str, ...]
    spool: BinaryIO
    dense: BinaryIO
    directory: Path
    max_temporary_bytes: int | None
    reserve_free_bytes: int = 0
    count: int = 0
    spooled_bytes: int = 0
    _mapping: mmap.mmap | None = field(default=None, init=False)

    def check_capacity(self, additional: int) -> None:
        """Admit logical scratch growth; concurrent disk consumption can still cause I/O errors."""
        dense_bytes = prod(self.shape) * 4
        needed = dense_bytes + self.spooled_bytes + additional
        if self.max_temporary_bytes is not None and needed > self.max_temporary_bytes:
            raise ValueError(
                f"voxel export temporary storage needs {needed} bytes, exceeding "
                f"the temporary budget of {self.max_temporary_bytes} bytes"
            )
        # The dense file is sparse until maps are expanded; reserve its full logical size.
        if (dense_bytes + additional + self.reserve_free_bytes
                > shutil.disk_usage(self.directory).free):
            raise OSError(errno.ENOSPC, "insufficient free space for voxel export scratch")

    def append(
        self,
        centers: NDArray[np.int32],
        features: Mapping[str, NDArray[np.float32]],
    ) -> None:
        """Append one evaluated batch, rejecting a budget violation before writing any rows.

        Coordinates are int32 ``(batch, ndim)`` and feature columns are float32
        ``(batch,)``. Coordinates are validated against the map shape and converted once
        per compact block to int64 row-major flat indices. An I/O failure invalidates the
        store and propagates to its owner.
        """
        count = len(centers)
        if centers.shape != (count, len(self.shape)) or centers.dtype != np.int32:
            raise ValueError("voxel centers must be int32 (batch, ndim)")
        if any(features[name].shape != (count,) or features[name].dtype != np.float32
               for name in self.names):
            raise ValueError("voxel features must be float32 (batch,) columns")
        additional = (
            count * (8 + 4 * len(self.names)) + 8 * ((count + 65535) // 65536)
        )
        self.check_capacity(additional)
        # Validate every block before the first write: an invalid late coordinate must
        # not leave earlier blocks that a subsequent map iteration would treat as
        # complete records. Blockwise min/max keeps the check bounded by block size.
        limits = np.asarray(self.shape, dtype=np.int64)
        for start in range(0, count, 65536):
            block = centers[start:start + 65536]
            if (block.min(axis=0) < 0).any() or (block.max(axis=0) >= limits).any():
                raise ValueError("voxel centers must lie inside the map shape")
        self.spool.seek(0, os.SEEK_END)
        for start in range(0, count, 65536):
            stop = min(start + 65536, count)
            block = centers[start:stop]
            np.array([stop - start], dtype=np.int64).tofile(self.spool)
            indices = np.ravel_multi_index(
                tuple(block[:, axis] for axis in range(len(self.shape))), self.shape,
            ).astype(np.int64, copy=False)
            indices.tofile(self.spool)
            for name in self.names:
                features[name][start:stop].tofile(self.spool)
        self.count += count
        self.spooled_bytes += additional

    def maps(self) -> Iterator[tuple[str, NDArray[np.float32]]]:
        """Yield borrowed disk-backed float32 maps, valid until the next map or context exit.

        Fresh sparse backing supplies zero background without touching the full grid.
        Only the selected value column and flat indices are read for each bounded block;
        NaN, infinity and signed zero are retained. Append every batch before iterating.
        """
        for index, name in enumerate(self.names):
            self.close()
            self.dense.truncate(0)
            self.dense.truncate(prod(self.shape) * 4)
            self._mapping = mmap.mmap(
                self.dense.fileno(), 0, access=mmap.ACCESS_WRITE, trackfd=False,
            )
            output = np.ndarray(self.shape, dtype=np.float32, buffer=self._mapping)
            flat = output.reshape(-1)
            position, remaining = 0, self.count
            while remaining:
                self.spool.seek(position)
                size = int(np.fromfile(self.spool, dtype=np.int64, count=1)[0])
                base = self.spool.tell()
                indices = np.fromfile(self.spool, dtype=np.int64, count=size)
                self.spool.seek(base + size * (8 + 4 * index))
                values = np.fromfile(self.spool, dtype=np.float32, count=size)
                flat[indices] = values
                position = base + size * (8 + 4 * len(self.names))
                remaining -= size
            # Consumers read this live mapping; scratch durability is unnecessary before discard.
            yield name, output

    def close(self) -> None:
        """Release the active mapping, including when an iterator was abandoned."""
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None


@contextmanager
def voxel_map_store(
    shape: tuple[int, ...], names: tuple[str, ...], *,
    temporary_dir: str | Path | None = None, max_temporary_bytes: int | None = None,
    reserve_free_bytes: int = 0,
) -> Generator[VoxelMapStore]:
    """Own two unlinked scratch files and one mapping; close them on failure or success.

    Logical scratch is ``4 * image voxels + ROI voxels * (8 + 4 * features)``
    plus 8 bytes per compact block of up to 65,536 rows. The optional budget limits
    that logical scratch, not output/publication files or operating-system page cache.
    A mapping can be resident in RSS; it is not a hard RAM cap. Scratch placement is
    explicit and does not change the process-wide temporary directory. ``reserve_free_bytes``
    leaves admission headroom for publication on a shared filesystem; it is outside the
    scratch budget and is not an operating-system reservation.
    """
    if len(shape) not in (2, 3) or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError("voxel map shape must be nonempty 2D or 3D")
    if not names or len(set(names)) != len(names):
        raise ValueError("voxel feature names must be nonempty and unique")
    if max_temporary_bytes is not None and (
        type(max_temporary_bytes) is not int or max_temporary_bytes <= 0
    ):
        raise ValueError("max_temporary_bytes must be a positive integer")
    if type(reserve_free_bytes) is not int or reserve_free_bytes < 0:
        raise ValueError("reserve_free_bytes must be a nonnegative integer")
    directory = (Path(temporary_dir).expanduser() if temporary_dir is not None
                 else Path(gettempdir()))
    if not directory.is_dir():
        raise ValueError(f"temporary_dir must be an existing directory: {directory}")
    with ExitStack() as resources:
        spool = resources.enter_context(TemporaryFile(dir=directory))
        dense = resources.enter_context(TemporaryFile(dir=directory))
        store = VoxelMapStore(shape, names, spool, dense, directory, max_temporary_bytes,
                              reserve_free_bytes=reserve_free_bytes)
        resources.callback(store.close)
        store.check_capacity(0)
        yield store
