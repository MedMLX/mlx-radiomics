"""Check published IBSI 1 and 2 benchmarks using the native extraction paths."""

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass, replace
from decimal import ROUND_DOWN, Decimal
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import urlopen
from xml.etree import ElementTree
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

from mlxradiomics import Settings, extract_files
from mlxradiomics.config.filtering import (
    FilterPadding,
    FourierGrid,
    LogKernel,
    OrientationPooling,
)
from mlxradiomics.config.settings import DirectionalAggregation, MatrixAggregation
from mlxradiomics.config.spatial import IbsiInterpolation, ImageSettings, Interpolator
from mlxradiomics.config.transforms import (
    GaborSettings,
    ImageType,
    LawsSettings,
    LoGSettings,
    MeanSettings,
    SeparableWaveletSettings,
    SimoncelliSettings,
    TransformSettings,
)
from mlxradiomics.filters.spectral import FFT_AVAILABLE
from mlxradiomics.imaging.io import read_image
from mlxradiomics.imaging.preprocessing import prepare_sampled_input
from mlxradiomics.imaging.spatial import ImagePair, load_image_pair
from mlxradiomics.imaging.transforms import feature_images
from mlxradiomics.numerics.runtime import mx
from scripts._ibsi_feature_tags import IBSI1_FEATURES

_DATA = "https://raw.githubusercontent.com/theibsi/data_sets/6da96021bc91faf4c0cb7fd7fa56a4225d2064a8"
_FILTERS = (
    "https://raw.githubusercontent.com/theibsi/ibsi_2_reference_data/"
    "5404579fb3e0d17e8db421f0e82d64ce2432ec03"
)
_WORKBOOK = "https://ibsi.radiomics.hevs.ch/assets/IBSI-1-submission-table.xlsx"
_WORKBOOK_SHA256 = "3af20577b580f2161bd546e147338ff226abb1676488e2906f2ce908b7985710"
_DIGITAL = Path("ibsi_1_digital_phantom/nifti")
_CT = Path("ibsi_1_ct_radiomics_phantom/nifti")
_IMAGE = Path("image/phantom.nii.gz")
_MASK = Path("mask/mask.nii.gz")
_FILTER_TABLE = Path("reference_feature_values/reference_values.csv")
_DIRECTIONS: dict[str, DirectionalAggregation] = {
    "2D_avg": "2d_average", "2D_comb": "2d_slice_merge",
    "2_5D_avg": "2.5d_direction_merge", "2_5D_comb": "2.5d_merge",
    "3D_avg": "3d_average", "3D_comb": "3d_merge",
}
_MATRICES: dict[str, MatrixAggregation] = {"2D": "2d_average", "2_5D": "2.5d_merge", "3D": "3d"}
_SUFFIX = re.compile(r"_(" + "|".join(sorted([*_DIRECTIONS, *_MATRICES], key=len, reverse=True))
                     + r")$")
_CONFIGS = {
    "Digital": (Settings(bin_width=1.0), (0.0, 0.0, 0.0), 482),
    "A": (Settings(resegment_range=(-500.0, 400.0)), (0.0, 0.0, 0.0), 346),
    "B": (Settings(bin_count=32, resegment_range=(-500.0, 400.0)), (0.0, 2.0, 2.0), 346),
    "C": (Settings(resegment_range=(-1000.0, 400.0), ivh_bin_width=2.5), (2.0, 2.0, 2.0), 210),
    "D": (Settings(bin_count=32, resegment_range=(-3.0, 3.0), resegment_mode="sigma"),
          (2.0, 2.0, 2.0), 210),
    "E": (Settings(bin_count=32, resegment_range=(-1000.0, 400.0), resegment_outlier_sigma=3.0,
                   ivh_bin_count=1000), (2.0, 2.0, 2.0), 210),
}
_ZERO = FilterPadding.CONSTANT
_MIRROR = FilterPadding.MIRROR
_AVERAGE = OrientationPooling.AVERAGE


@dataclass(frozen=True)
class Reference:
    tag: str
    value: Decimal
    tolerance: Decimal

    def check(self, actual: float, case: str, *, workbook: bool = False) -> int:
        if math.isfinite(actual):
            error = abs(Decimal(str(actual)) - self.value)
            if workbook:
                # Column H truncates the absolute difference to the precision of
                # three significant digits of the reference, with a unit step at zero.
                step = Decimal(1).scaleb(self.value.adjusted() - 2) if self.value else Decimal(1)
                error = error.quantize(step, rounding=ROUND_DOWN)
            if error <= self.tolerance:
                return 0
        print(f"FAIL {case} {self.tag}: {actual} vs {self.value} ± {self.tolerance}")
        return 1


def workbook_rows(path: Path, sheet: int) -> tuple[list[Reference], list[Reference]]:
    """Read the feature and diagnostic rows from the unchanged IBSI workbook."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != _WORKBOOK_SHA256:
        raise ValueError(f"Unexpected IBSI 1 workbook digest for {path}")
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as book:
        shared = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
        strings = ["".join(t.text or "" for t in node.findall(".//x:t", ns)) for node in shared]
        tree = ElementTree.fromstring(book.read(f"xl/worksheets/sheet{sheet}.xml"))
    features: list[Reference] = []
    diagnostics: list[Reference] = []
    for row in tree.findall("x:sheetData/x:row", ns)[1:]:
        values: dict[str, str] = {}
        for cell in row:
            value = cell.findtext("x:v", namespaces=ns)
            if value:
                key = cell.attrib["r"].rstrip("0123456789")
                values[key] = strings[int(value)] if cell.get("t") == "s" else value
        if "E" in values and values.get("D") != "not standardized":
            group = diagnostics if values["B"].startswith("Diagnostics") else features
            group.append(Reference(values["J"], Decimal(values["E"]), Decimal(values["F"])))
    return features, diagnostics


def spatial(
    spacing: tuple[float, ...], interpolator: Interpolator = Interpolator.LINEAR,
) -> ImageSettings:
    return ImageSettings(ibsi_interpolation=IbsiInterpolation(
        spacing=spacing, image_interpolator=interpolator,
    ))


def settings_for_aggregation(settings: Settings, suffix: str) -> Settings:
    if suffix in _DIRECTIONS:
        mode = _DIRECTIONS[suffix]
        return replace(settings, glcm_aggregation=mode, glrlm_aggregation=mode)
    if suffix:
        matrix = _MATRICES[suffix]
        return replace(settings, glszm_aggregation=matrix, gldzm_aggregation=matrix,
                       ngtdm_aggregation=matrix, gldm_aggregation=matrix)
    return settings


def image_diagnostics(
    metrics: dict[str, float], pair: ImagePair, suffix: str,
) -> None:
    for axis, dim, spacing in zip(
        "xyz", reversed(pair.image.array.shape), reversed(pair.image.spacing), strict=True,
    ):
        metrics[f"img_dim_{axis}_{suffix}"] = dim
        metrics[f"vox_dim_{axis}_{suffix}"] = spacing
    image = np.asarray(pair.image.array, dtype=np.float64)
    metrics[f"mean_int_{suffix}"] = float(np.mean(image))
    metrics[f"min_int_{suffix}"] = float(np.min(image))
    metrics[f"max_int_{suffix}"] = float(np.max(image))


def roi_diagnostics(
    metrics: dict[str, float],
    image: NDArray[np.generic],
    morphology: NDArray[np.bool_],
    intensity: NDArray[np.bool_],
    suffix: str,
) -> None:
    for kind, mask in (("int", intensity), ("morph", morphology)):
        coords = np.nonzero(mask)
        for axis, dim, indices in zip(
            "xyz", reversed(mask.shape), reversed(coords), strict=True,
        ):
            metrics[f"{kind}_mask_dim_{axis}_{suffix}"] = dim
            metrics[f"{kind}_mask_bb_dim_{axis}_{suffix}"] = int(np.ptp(indices)) + 1
        metrics[f"{kind}_mask_vox_count_{suffix}"] = coords[0].size
    selected = np.asarray(image[intensity], dtype=np.float64)
    metrics[f"int_mask_mean_int_{suffix}"] = float(np.mean(selected))
    metrics[f"int_mask_min_int_{suffix}"] = float(np.min(selected))
    metrics[f"int_mask_max_int_{suffix}"] = float(np.max(selected))


def check_diagnostics(
    source: Path, settings: Settings, options: ImageSettings, rows: list[Reference], case: str,
) -> int:
    metrics: dict[str, float] = {}
    original = load_image_pair(source / _IMAGE, source / _MASK,
                               image_settings=spatial((0.0, 0.0, 0.0)))
    interpolated = load_image_pair(source / _IMAGE, source / _MASK, image_settings=options)
    for suffix, pair in (("init", original), ("interp", interpolated)):
        mask = pair.mask.array == settings.label
        image_diagnostics(metrics, pair, f"{suffix}_img")
        roi_diagnostics(metrics, pair.image.array, mask, mask, f"{suffix}_roi")
    prepared, _ = prepare_sampled_input(
        interpolated.image.array, interpolated.mask.array, settings, voxel=False,
    )
    roi_diagnostics(metrics, prepared.image, prepared.shape_roi, prepared.roi, "reseg_roi")
    return sum(row.check(metrics[row.tag], case, workbook=True) for row in rows)


def chapter_one(root: Path) -> int:
    failures = 0
    for sheet, (case, (base, spacing, expected)) in enumerate(_CONFIGS.items(), start=1):
        rows, diagnostics = workbook_rows(root / "workbook.xlsx", sheet)
        if len(rows) != expected or len(diagnostics) != (0 if case == "Digital" else 60):
            raise ValueError(f"Unexpected IBSI 1 row counts for {case}")
        groups: dict[str, list[tuple[Reference, str]]] = {}
        for row in rows:
            match = _SUFFIX.search(row.tag)
            suffix = match.group(1) if match else ""
            tag = row.tag[:match.start()] if match else row.tag
            groups.setdefault(suffix, []).append((row, IBSI1_FEATURES[tag]))
        source = root / (_DIGITAL if case == "Digital" else _CT)
        options = spatial(spacing, Interpolator.BSPLINE if case == "E" else Interpolator.LINEAR)
        case_failures = 0
        for suffix, group in groups.items():
            settings = replace(settings_for_aggregation(base, suffix),
                               features=tuple(feature for _, feature in group))
            result = extract_files(source / _IMAGE, source / _MASK,
                                   settings=settings, image_settings=options)
            case_failures += sum(row.check(result.features[f"original_{feature}"], case,
                                           workbook=True) for row, feature in group)
        if diagnostics:
            case_failures += check_diagnostics(source, base, options, diagnostics, case)
        print(f"IBSI 1 {case}: {len(rows)} features, {len(diagnostics)} diagnostics, "
              f"{case_failures} failures", flush=True)
        failures += case_failures
    return failures


def gabor(
    sigma: float, wavelength: float, gamma: float, theta: tuple[float, ...],
    padding: FilterPadding, axial: bool,
) -> GaborSettings:
    return GaborSettings(
        sigma=(sigma,), wavelength=(wavelength,), gamma=gamma, theta=theta, padding=padding,
        slice_axis=0 if axial else None,
        pooling=_AVERAGE if len(theta) > 1 else OrientationPooling.NONE,
    )


def angles(count: int) -> tuple[float, ...]:
    # The modulus is invariant under a half-turn (conjugating the complex response).
    return tuple(i * math.pi / count for i in range(count))


def map_cases() -> list[tuple[str, str, TransformSettings]]:
    """IBSI 2 manual v9, Table 6.1, for maps with standardized references."""
    result: list[tuple[str, str, TransformSettings]] = []
    padding = (_ZERO, FilterPadding.NEAREST, FilterPadding.PERIODIC, _MIRROR)
    for index, pad in enumerate(padding, start=1):
        result.append((f"1_a_{index}", "checkerboard", TransformSettings(
            image_types=(ImageType.MEAN,), mean=MeanSettings(15, pad),
        )))
    result.append(("1_b_1", "impulse", TransformSettings(
        image_types=(ImageType.MEAN,), mean=MeanSettings(15, _ZERO, planar=True),
    )))
    for key, phantom, sigma, pad, planar in (
        ("2_a", "impulse", 3.0, _ZERO, False),
        ("2_b", "checkerboard", 5.0, _MIRROR, False),
        ("2_c", "checkerboard", 5.0, _MIRROR, True),
    ):
        result.append((key, phantom, TransformSettings(
            image_types=(ImageType.LOG,), log=LoGSettings(
                (sigma,), LogKernel.SAMPLED, padding=pad, planar=planar,
            ),
        )))
    for key, phantom, kernel, pad in (
        ("a", "impulse", "E5L5S5", _ZERO),
        ("b", "checkerboard", "E3W5R5", _MIRROR),
        ("c", "checkerboard", "L5S5", _MIRROR),
    ):
        for index in range(1, 4):
            result.append((f"3_{key}_{index}", phantom, TransformSettings(
                image_types=(ImageType.LAWS,), laws=LawsSettings(
                    kernel, pad, pooling=OrientationPooling.NONE if index == 1
                    else OrientationPooling.MAX, energy_distance=7 if index == 3 else None,
                ),
            )))
    for key, phantom, options in (
        ("4_a_1", "impulse", gabor(10, 4, 0.5, (math.pi / 3,), _ZERO, True)),
        ("4_a_2", "impulse", gabor(10, 4, 0.5, angles(4), _ZERO, False)),
        ("4_b_1", "sphere", gabor(20, 8, 2.5, (5 * math.pi / 4,), _MIRROR, True)),
        ("4_b_2", "sphere", gabor(20, 8, 2.5, angles(8), _MIRROR, False)),
    ):
        result.append((key, phantom, TransformSettings(image_types=(ImageType.GABOR,),
                                                       gabor=options)))
    for key, phantom, wavelet, band, level, pad, pool in (
        ("5_a_1", "impulse", "db2", "LHL", 1, _ZERO, OrientationPooling.NONE),
        ("5_a_2", "impulse", "db2", "LHL", 1, _ZERO, _AVERAGE),
        ("6_a_1", "sphere", "coif1", "HHL", 1, FilterPadding.PERIODIC, OrientationPooling.NONE),
        ("6_a_2", "sphere", "coif1", "HHL", 1, FilterPadding.PERIODIC, _AVERAGE),
        ("7_a_1", "checkerboard", "haar", "LLL", 2, _MIRROR, _AVERAGE),
        ("7_a_2", "checkerboard", "haar", "HHH", 2, _MIRROR, _AVERAGE),
    ):
        result.append((key, phantom, TransformSettings(
            image_types=(ImageType.SEPARABLE_WAVELET,), separable_wavelet=SeparableWaveletSettings(
                wavelet, (band,), level, padding=pad, pooling=pool,
            ),
        )))
    for level in (1, 2, 3):
        result.append((f"8_a_{level}", "checkerboard", TransformSettings(
            image_types=(ImageType.SIMONCELLI,),
            simoncelli=SimoncelliSettings(levels=(level,), grid=FourierGrid.CENTERED),
        )))
    return result


def map_reference(case: str) -> Path:
    return Path(f"reference_response_maps/{case}-ValidCRM.nii")


def map_input(phantom: str) -> Path:
    return Path(f"ibsi_2_digital_phantom/nifti/{phantom}/image/{phantom}.nii.gz")


def check_maps(root: Path, cases: list[tuple[str, str, TransformSettings]]) -> int:
    checked = failures = 0
    for case, phantom, settings in cases:
        if ImageType.SIMONCELLI in settings.image_types and not FFT_AVAILABLE:
            continue
        image = read_image(root / map_input(phantom))
        reference = np.asarray(read_image(root / map_reference(case)).array, dtype=np.float64)
        with mx.stream(mx.gpu):
            pixels = mx.array(np.asarray(image.array, dtype=np.float32))
            [response] = feature_images(pixels, image.spacing, settings)
            values = np.asarray(response.image, dtype=np.float64)
        if values.shape != reference.shape:
            raise ValueError(f"{case}: response shape {values.shape} vs {reference.shape}")
        error = float(np.max(np.abs(values - reference)))
        tol = 0.01 * float(np.ptp(reference))
        if not math.isfinite(error) or error > tol:
            print(f"FAIL IBSI 2 map {case}: max error {error} exceeds {tol}")
            failures += 1
        checked += 1
    print(f"IBSI 2 phantom: {checked} maps, {len(cases) - checked} FFT-gated, "
          f"{failures} failures", flush=True)
    return failures


def ct_filters(planar: bool) -> tuple[TransformSettings, ...]:
    """IBSI 2 manual v9, Table 6.3, for configurations with reference values."""
    return (
        TransformSettings(),
        TransformSettings(image_types=(ImageType.MEAN,), mean=MeanSettings(5, planar=planar)),
        TransformSettings(image_types=(ImageType.LOG,),
                          log=LoGSettings((1.5,), LogKernel.SAMPLED, planar=planar)),
        TransformSettings(image_types=(ImageType.LAWS,), laws=LawsSettings(
            "L5E5" if planar else "L5E5E5", pooling=OrientationPooling.MAX, energy_distance=7,
        )),
        TransformSettings(image_types=(ImageType.GABOR,),
                          gabor=gabor(5, 2, 1.5, angles(8), _MIRROR, planar)),
        TransformSettings(
            image_types=(ImageType.SEPARABLE_WAVELET,), separable_wavelet=SeparableWaveletSettings(
                "db3", ("LH" if planar else "LLH",), pooling=_AVERAGE, planar=planar,
            ),
        ),
        TransformSettings(
            image_types=(ImageType.SEPARABLE_WAVELET,), separable_wavelet=SeparableWaveletSettings(
                "db3", ("HH" if planar else "HHH",), level=2, pooling=_AVERAGE, planar=planar,
            ),
        ),
        *(TransformSettings(
            image_types=(ImageType.SIMONCELLI,), simoncelli=SimoncelliSettings(
                levels=(level,), grid=FourierGrid.CENTERED, planar=planar,
            ),
        ) for level in (1, 2)),
    )


def check_filtered_ct(root: Path) -> int:
    cases: dict[str, list[Reference]] = {}
    with (root / _FILTER_TABLE).open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row["consensus_value"]:
                cases.setdefault(row["filter_id"], []).append(Reference(
                    row["feature_tag"], Decimal(row["consensus_value"]), Decimal(row["tolerance"]),
                ))
    checked = skipped = failures = 0
    for letter in ("A", "B"):
        planar = letter == "A"
        options = spatial((0.0, 0.0, 0.0)) if planar else spatial(
            (1.0, 1.0, 1.0), Interpolator.BSPLINE,
        )
        for index, transforms in enumerate(ct_filters(planar), start=1):
            case = f"{index}.{letter}"
            rows = cases[case]
            if len(rows) != (17 if case == "8.B" else 18):
                raise ValueError(f"Unexpected IBSI 2 CT row count for {case}")
            if ImageType.SIMONCELLI in transforms.image_types and not FFT_AVAILABLE:
                skipped += len(rows)
                continue
            features = tuple(IBSI1_FEATURES[row.tag] for row in rows)
            result = extract_files(
                root / _CT / _IMAGE, root / _CT / _MASK,
                settings=Settings(features=features, resegment_range=(-1000.0, 400.0)),
                image_settings=options, transforms=transforms,
            )
            for row, feature in zip(rows, features, strict=True):
                value, = (v for key, v in result.features.items() if key.endswith(f"_{feature}"))
                failures += row.check(value, f"IBSI 2 CT {case}")
            checked += len(rows)
    print(f"IBSI 2 chest CT: {checked} features, {skipped} FFT-gated, {failures} failures",
          flush=True)
    return failures


def download(root: Path, cases: list[tuple[str, str, TransformSettings]]) -> None:
    """Fetch the pinned public data into the caller's cache directory."""
    inputs = {_DIGITAL / _IMAGE, _DIGITAL / _MASK, _CT / _IMAGE, _CT / _MASK}
    inputs.update(map_input(phantom) for _, phantom, _ in cases)
    references = {_FILTER_TABLE, *(map_reference(case) for case, _, _ in cases)}
    files = {path: f"{_DATA}/{path}" for path in inputs}
    files.update({path: f"{_FILTERS}/{path}" for path in references})
    files[Path("workbook.xlsx")] = _WORKBOOK
    for file, url in sorted(files.items()):
        target = root / file
        if target.exists():
            continue
        with urlopen(url, timeout=30) as response:
            contents = response.read()
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=target.parent) as staging:
            staged = Path(staging) / target.name
            staged.write_bytes(contents)
            staged.replace(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="IBSI dataset cache directory")
    parser.add_argument("--download", action="store_true", help="fetch missing pinned input files")
    args = parser.parse_args()
    root = Path(args.data)
    cases = map_cases()
    if args.download:
        download(root, cases)
    print(f"MLX {version('mlx')} on {mx.device_info(mx.gpu)['device_name']}", flush=True)
    return int(bool(chapter_one(root) + check_maps(root, cases) + check_filtered_ct(root)))


if __name__ == "__main__":
    raise SystemExit(main())
