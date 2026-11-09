"""Private output staging shared by the MedMLX runner and the public exporters.

The runner is :func:`mlxradiomics.medmlx.run`.

All staged files stay on the destination filesystem so publication is a rename or a
no-clobber link. Callers own the output directory lock and decide which targets may be
replaced; :class:`Publication` only stages files and restores replaced targets if a
later publication step fails.
"""

from __future__ import annotations

import dataclasses
import fcntl
import os
import re
import shutil
import signal
import tempfile
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from threading import current_thread, main_thread
from types import FrameType

from medmlx_core.errors import InvalidInputError, ModelExecutionError

from mlxradiomics.imaging.io import has_detached_data

_STAGING_NAME = re.compile(r"\.mlxradiomics-[a-z0-9_]{8}")
_MARKER = ".mlxradiomics-staging"


@contextmanager
def _termination_checkpoints() -> Generator[Callable[[], None]]:
    """Defer default main-thread SIGTERM until transaction cleanup has completed.

    Checkpoints interrupt only between consistent filesystem operations. The handler
    stays non-raising during rollback and cleanup, including repeated SIGTERM. Host
    applications retain custom/ignored handlers and signal ownership in threaded calls.
    """
    termination_requested = False

    def request_termination(_signum: int, _frame: FrameType | None) -> None:
        nonlocal termination_requested
        termination_requested = True

    def checkpoint() -> None:
        if termination_requested:
            raise SystemExit(128 + signal.SIGTERM)

    owned = (
        current_thread() is main_thread()
        and signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    )
    try:
        if owned:
            signal.signal(signal.SIGTERM, request_termination)
        yield checkpoint
    finally:
        if owned and signal.getsignal(signal.SIGTERM) is request_termination:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        if termination_requested:
            os.kill(os.getpid(), signal.SIGTERM)
            raise SystemExit(128 + signal.SIGTERM)


@contextmanager
def output_lock(directory: Path) -> Generator[None]:
    """Refuse overlapping runs using the same directory, without a persistent lock file."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    except OSError as error:
        raise InvalidInputError(f"invalid output directory: {directory}") from error
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise InvalidInputError(
                f"another run is using output directory: {directory}"
            ) from error
        yield
    finally:
        # Closing the descriptor also releases the lock after failure or cancellation.
        os.close(descriptor)


def reject_input_aliases(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
    """Refuse output paths resolving to, or hard-linked to, an input file."""
    for output in outputs:
        for source in inputs:
            if output.resolve() == source.resolve() or (
                output.exists() and output.samefile(source)
            ):
                raise InvalidInputError(f"output aliases an input: {output}")


def reject_output_aliases(outputs: tuple[Path, ...]) -> None:
    """Refuse outputs sharing a resolved path or an existing file identity."""
    resolved: set[Path] = set()
    identities: set[tuple[int, int]] = set()
    for output in outputs:
        path = output.resolve()
        if path in resolved:
            raise InvalidInputError(f"output aliases another output: {output}")
        resolved.add(path)
        if output.exists():
            info = output.stat()
            identity = (info.st_dev, info.st_ino)
            if identity in identities:
                raise InvalidInputError(f"output aliases another output: {output}")
            identities.add(identity)


def prepare_output_targets(directory: Path, outputs: tuple[Path, ...]) -> None:
    """Reject non-file targets and linked map directories before extraction."""
    for path in outputs:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise InvalidInputError(f"output target must be a regular file: {path}")
        if path.parent != directory and path.parent.is_symlink():
            raise InvalidInputError(f"output parent must not be a symbolic link: {path.parent}")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise InvalidInputError(f"invalid output parent: {path.parent}") from error


def check_output_replacement(
    existing: list[str], *, overwrite: bool, inputs: tuple[Path, ...],
) -> None:
    """Refuse or admit replacement, including detached input payloads."""
    if not existing:
        return
    if not overwrite:
        raise InvalidInputError(
            f"outputs already exist: {', '.join(existing)}; pass overwrite=True to replace them"
        )
    # A detached image can reference output files or their hard links. Refuse
    # all replacement rather than guessing every possible payload path.
    if any(has_detached_data(source) for source in inputs):
        raise InvalidInputError(
            "cannot replace existing outputs with detached NRRD, NIfTI or Analyze inputs; "
            "choose a new output directory"
        )


@dataclasses.dataclass
class Publication:
    """Own exclusive staging files and restore replaced outputs after a publication failure."""

    root: Path
    staged: dict[Path, Path] = dataclasses.field(default_factory=dict[Path, Path])
    retain_backups: bool = False
    _cleaned: bool = dataclasses.field(default=False, init=False)

    @classmethod
    def create(cls, directory: Path) -> Publication:
        """Create a marked private staging directory on the output filesystem."""
        root = Path(tempfile.mkdtemp(prefix=".mlxradiomics-", dir=directory))
        (root / _MARKER).write_text("1\n", encoding="ascii")
        return cls(root)

    def stage(self, target: Path) -> Path:
        """Reserve a private file with the target's suffix for format-selecting writers."""
        descriptor, filename = tempfile.mkstemp(dir=self.root, suffix="".join(target.suffixes))
        os.close(descriptor)
        temporary = Path(filename)
        self.staged[target] = temporary
        return temporary

    def publish(self, *, overwrite: bool) -> None:
        """Publish or restore a generation, then clean staging before normal termination.

        Main-thread calls defer default SIGTERM through rollback and cleanup. SIGKILL
        and host-defined signal policies retain the process owner's semantics. Readers
        can still observe individual file replacements while publication is running.
        """
        with _termination_checkpoints() as checkpoint:
            try:
                backups: dict[Path, Path] = {}
                attempted: list[tuple[Path, os.stat_result]] = []
                complete = False
                try:
                    # Keep recovery copies if rollback is interrupted by host policy.
                    self.retain_backups = True
                    if overwrite:
                        for target in self.staged:
                            checkpoint()
                            if target.exists():
                                backup = self.root / f"backup-{target.name}"
                                # Preserve the old inode without copying full voxel maps.
                                os.link(target, backup, follow_symlinks=False)
                                backups[target] = backup
                    for target, temporary in self.staged.items():
                        checkpoint()
                        # Record ownership before the syscall: SIGINT can interrupt its return.
                        attempted.append((target, temporary.stat()))
                        if overwrite:
                            os.replace(temporary, target)
                        else:
                            try:
                                os.link(temporary, target)
                            except FileExistsError as error:
                                raise InvalidInputError(
                                    f"output already exists: {target}; "
                                    "pass overwrite=True to replace it"
                                ) from error
                        checkpoint()
                    complete = True
                finally:
                    if not complete:
                        failures: list[OSError] = []
                        for target, staged_info in reversed(attempted):
                            backup = backups.get(target)
                            try:
                                try:
                                    current = target.stat(follow_symlinks=False)
                                except FileNotFoundError:
                                    if backup is not None:
                                        os.link(backup, target, follow_symlinks=False)
                                    continue
                                if not os.path.samestat(current, staged_info):
                                    # A failed no-clobber link must preserve another writer's file.
                                    if backup is not None and not os.path.samestat(
                                        current, backup.stat(),
                                    ):
                                        raise OSError(f"output changed during rollback: {target}")
                                    continue
                                if backup is not None:
                                    os.replace(backup, target)
                                else:
                                    target.unlink()
                            except OSError as error:
                                failures.append(error)
                        if failures:
                            raise ModelExecutionError(
                                f"cannot restore outputs; previous files retained at {self.root}"
                            ) from failures[0]
                    self.retain_backups = False
            finally:
                self.cleanup()

    def cleanup(self) -> None:
        """Clean owned staging once; retain recovery copies if rollback failed."""
        if not self._cleaned and not self.retain_backups:
            # Python 3.13 only propagates a missing root, including a cleanup retry.
            with suppress(FileNotFoundError):
                shutil.rmtree(self.root)
            self._cleaned = True


def reclaim_stale_staging(directory: Path) -> None:
    """Remove this library's abandoned staging after taking the directory lock.

    A prior process that was killed cannot run Python cleanup. Its marked staging
    directory is therefore reclaimed by the next run, which holds the exclusive lock.
    Directories containing a ``backup-*`` recovery copy are always retained. Empty
    name-matching directories are removed; unrecognized nonempty directories are kept.
    Reclamation is best effort: an unreadable directory or entry is skipped and a
    removal failure is ignored, since abandoned staging never affects correctness.
    """
    try:
        entries = tuple(directory.iterdir())
    except OSError:
        return
    for candidate in entries:
        if not _STAGING_NAME.fullmatch(candidate.name) or candidate.is_symlink():
            continue
        if not candidate.is_dir():
            continue
        try:
            contents = tuple(candidate.iterdir())
        except OSError:
            continue
        if any(path.name.startswith("backup-") for path in contents):
            continue
        marked = any(path.name == _MARKER for path in contents)
        if marked or not contents:
            shutil.rmtree(candidate, ignore_errors=True)
