"""Admission for selected ROI count buffers before their device allocation."""

from dataclasses import replace
from math import prod

from mlxradiomics.config.settings import Settings
from mlxradiomics.features.aggregation import resolved_aggregations
from mlxradiomics.features.selection import explicit_texture
from mlxradiomics.features.sparse_glcm import use_sparse_glcm
from mlxradiomics.imaging.preprocessing import directions


def check_roi_count_buffers(shape: tuple[int, ...], ng: int, settings: Settings) -> None:
    """Bound primary dense count arrays, retaining the requested discretization.

    ``shape`` is the actual padded ROI window, not the original file grid. The
    scratch budget bounds each selected primary count buffer, not the combined
    live inputs, casts or statistic intermediates. NGTDM partial tables may also
    scale linearly with their input domain, as their existing tiled reduction does.
    """
    names = set(settings.features)
    budget = settings.scratch_mib * 1024**2

    def admit(family: str, cells: int, itemsize: int, *, index_bits: int = 32,
              allowance: int = budget) -> None:
        if cells >= 2**index_bits:
            raise MemoryError(
                f"{family} ROI count buffer exceeds {index_bits}-bit indexing capacity"
            )
        size = cells * itemsize
        if size > allowance:
            raise MemoryError(
                f"{family} ROI count buffer requires {size} bytes; "
                f"scratch_mib={settings.scratch_mib} permits {allowance} bytes"
            )

    if names & {"firstorder_Entropy", "firstorder_Uniformity"}:
        admit("firstorder", ng, 4, index_bits=31)
    if any(name.startswith("intensityhistogram_") for name in names):
        admit("intensityhistogram", ng, 8, index_bits=31)
    if use_sparse_glcm(ng):
        if names & {"glcm_Imc1", "glcm_Imc2", "glcm_Correlation", "glcm_DifferenceEntropy"}:
            admit("glcm", ng, 4, index_bits=31)
        if names & {"glcm_SumEntropy", "glcm_ClusterShade"}:
            admit("glcm", 2 * ng - 1, 4, index_bits=31)

    explicit = explicit_texture(settings)
    modes = resolved_aggregations(settings, len(shape)) if explicit else {}
    for family in ("glrlm", "gldm", "ngtdm"):
        if not any(name.startswith(family + "_") for name in names):
            continue
        domain, effective = shape, settings
        if explicit and len(shape) == 3 and not modes[family].startswith("3d"):
            axis = settings.slice_axis if settings.force2d_axis is None else settings.force2d_axis
            domain = tuple(size for index, size in enumerate(shape) if index != axis)
            effective = replace(settings, force2d_axis=None)
        if family == "glrlm":
            admit(family, ng * max(domain), 4)
        elif family == "gldm":
            if effective.neighborhood_support == "closed-radius":
                effective = replace(
                    effective, distances=tuple(range(1, effective.neighborhood_radius + 1)),
                )
            columns = 2 * len(directions(domain, effective)) + 1
            admit(family, ng * columns, 4)
        else:
            width = prod(domain)
            if not explicit and width > 64:
                # Native ROI sums use one int32 and one float32 array of Ng values.
                admit(family, ng, 4, index_bits=31)
            else:
                chunks = max(1, min((width + 2047) // 2048, width // ng))
                admit(family, ng * chunks, 4, allowance=max(budget, width * 4))
                admit(family, ng, 8, index_bits=31)
