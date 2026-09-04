import math
from dataclasses import replace

import numpy as np
import pytest
from tests.support import FakeArray

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ContractError
from femx.interop.fdtdx import (
    FDTDXFingerprint,
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SampledVectorField,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeGrid,
    YeeVectorField,
)

pytestmark = pytest.mark.unit

SHA = "a" * 64


def yee_grid(*, shape: tuple[int, int, int] = (4, 5, 1)) -> YeeGrid:
    return YeeGrid(
        edge_coordinates=tuple(FakeArray((size + 1,)) for size in shape),
        coordinate_sha256=SHA,
    )


def yee_field(kind: YeeFieldKind, unit: str, *, grid: YeeGrid | None = None) -> YeeVectorField:
    resolved_grid = yee_grid() if grid is None else grid
    return YeeVectorField(
        values=FakeArray((3, *resolved_grid.shape), dtype_kind="c"),
        grid=resolved_grid,
        field_kind=kind,
        unit=unit,
    )


def transfer_report() -> TransferReport:
    fdtdx = FDTDXFingerprint("0.6.2", "b" * 40, "c" * 64)
    source_power = 1.0
    pre_correction_power = 1.1
    correction_scale = math.sqrt(source_power / pre_correction_power)
    return TransferReport(
        source_representation=FieldRepresentation.FEM_DOFS,
        target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
        operator_sha256=SHA,
        relative_power_error=0.0,
        source_power_watts=source_power,
        pre_correction_power_watts=pre_correction_power,
        relative_pre_correction_power_error=0.1,
        transferred_power_watts=source_power,
        power_correction_scale=correction_scale,
        target_runtime_name="fdtdx",
        target_runtime_version=fdtdx.package_version,
        target_source_revision=fdtdx.source_revision,
        target_source_digest=fdtdx.source_digest,
    )


def mode_bundle_kwargs() -> dict[str, object]:
    grid = yee_grid()
    return {
        "frequency_hz": 193.4e12,
        "effective_index": 2.41 + 0j,
        "beta_per_m": 9.8e6 + 0j,
        "electric": yee_field(YeeFieldKind.ELECTRIC, "V/m", grid=grid),
        "magnetic": yee_field(YeeFieldKind.MAGNETIC, "V/m", grid=grid),
        "propagation": AxisDirection(Axis.Z, Direction.POSITIVE),
        "magnetic_convention": MagneticFieldConvention.ETA0_H,
        "normalization": ModeNormalization(target_power_watts=1.0),
        "solver": SolverFingerprint("elmer", "26.2", SHA, SHA, "4f2d7e4"),
        "transfer": transfer_report(),
    }


def test_mode_bundle_preserves_fdtdx_boundary_conventions() -> None:
    grid = yee_grid()
    transfer = transfer_report()
    bundle = ModeBundle(
        frequency_hz=193.4e12,
        effective_index=2.41 + 0j,
        beta_per_m=9.8e6 + 0j,
        electric=yee_field(YeeFieldKind.ELECTRIC, "V/m", grid=grid),
        magnetic=yee_field(YeeFieldKind.MAGNETIC, "V/m", grid=grid),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=1.0),
        solver=SolverFingerprint("elmer", "26.2", SHA, SHA, "4f2d7e4"),
        transfer=transfer,
    )

    assert bundle.magnetic_convention is MagneticFieldConvention.ETA0_H
    assert bundle.transfer is not None
    assert bundle.transfer.relative_power_error == 0.0
    assert bundle.electric.spatial_offsets == (
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
    )
    assert bundle.magnetic.spatial_offsets == (
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.0),
    )


def test_mode_bundle_rejects_shape_dtype_and_unit_ambiguity() -> None:
    with pytest.raises(ContractError, match="complex dtype"):
        SampledVectorField(
            FakeArray((3, 2), dtype_kind="f"),
            (FakeArray((2,)),),
            FieldRepresentation.CARTESIAN_SAMPLES,
            "V/m",
            "samples",
        )
    with pytest.raises(ContractError, match="coordinate lengths"):
        SampledVectorField(
            FakeArray((3, 2), dtype_kind="c"),
            (FakeArray((3,)),),
            FieldRepresentation.CARTESIAN_SAMPLES,
            "V/m",
            "samples",
        )

    with pytest.raises(ContractError, match="requires magnetic unit"):
        grid = yee_grid(shape=(1, 4, 5))
        ModeBundle(
            1.0,
            1 + 0j,
            1 + 0j,
            yee_field(YeeFieldKind.ELECTRIC, "V/m", grid=grid),
            yee_field(YeeFieldKind.MAGNETIC, "A/m", grid=grid),
            AxisDirection(Axis.X),
            MagneticFieldConvention.ETA0_H,
            ModeNormalization(),
            SolverFingerprint("solver", "1", SHA, SHA),
            transfer_report(),
        )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: SampledVectorField(
                FakeArray((2, 4), dtype_kind="c"),
                (FakeArray((4,)),),
                FieldRepresentation.CARTESIAN_SAMPLES,
                "V/m",
                "samples",
            ),
            "three components",
        ),
        (
            lambda: SampledVectorField(
                FakeArray((3, 4), dtype_kind="c"),
                (),
                FieldRepresentation.CARTESIAN_SAMPLES,
                "V/m",
                "samples",
            ),
            "physical coordinates",
        ),
        (
            lambda: SampledVectorField(
                FakeArray((3, 4), dtype_kind="c"),
                (FakeArray((4,)),),
                FieldRepresentation.FEM_DOFS,
                "V/m",
                "Hcurl",
            ),
            "Cartesian samples",
        ),
        (
            lambda: SampledVectorField(
                FakeArray((3, 4), dtype_kind="c"),
                (FakeArray((2, 2)),),
                FieldRepresentation.CARTESIAN_SAMPLES,
                "V/m",
                "samples",
            ),
            "one-dimensional",
        ),
        (
            lambda: SampledVectorField(
                FakeArray((3, 4), dtype_kind="c"),
                (FakeArray((4,)),),
                FieldRepresentation.CARTESIAN_SAMPLES,
                "",
                "samples",
            ),
            "declare unit",
        ),
        (lambda: ModeNormalization(target_power_watts=0), "power must be positive"),
        (lambda: ModeNormalization(phase_reference=""), "phase reference"),
        (lambda: SolverFingerprint("", "1", SHA, SHA), "name and version"),
        (lambda: SolverFingerprint("solver", "1", "bad", SHA), "config_sha256"),
        (
            lambda: TransferReport(
                FieldRepresentation.FEM_DOFS,
                FieldRepresentation.CARTESIAN_SAMPLES,
                "bad",
                0,
            ),
            "operator digest",
        ),
        (
            lambda: TransferReport(
                FieldRepresentation.FEM_DOFS,
                FieldRepresentation.CARTESIAN_SAMPLES,
                SHA,
                -1,
            ),
            "power error",
        ),
        (
            lambda: TransferReport(
                FieldRepresentation.FEM_DOFS,
                FieldRepresentation.CARTESIAN_SAMPLES,
                SHA,
                0,
                -1,
            ),
            "interpolation error",
        ),
    ],
)
def test_mode_contract_rejects_missing_conventions(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


def test_mode_bundle_rejects_frequency_field_and_schema_mismatches() -> None:
    grid = yee_grid()
    common = {
        "effective_index": 2 + 0j,
        "beta_per_m": 1 + 0j,
        "electric": yee_field(YeeFieldKind.ELECTRIC, "V/m", grid=grid),
        "magnetic": yee_field(YeeFieldKind.MAGNETIC, "A/m", grid=grid),
        "propagation": AxisDirection(Axis.Z),
        "magnetic_convention": MagneticFieldConvention.PHYSICAL_H,
        "normalization": ModeNormalization(),
        "solver": SolverFingerprint("solver", "1", SHA, SHA),
        "transfer": transfer_report(),
    }
    with pytest.raises(ContractError, match="frequency"):
        ModeBundle(frequency_hz=0, **common)

    mismatched = yee_field(YeeFieldKind.MAGNETIC, "A/m", grid=yee_grid(shape=(2, 2, 1)))
    with pytest.raises(ContractError, match="different Yee grids"):
        ModeBundle(frequency_hz=1, **{**common, "magnetic": mismatched})
    with pytest.raises(ContractError, match="unsupported mode schema"):
        ModeBundle(frequency_hz=1, **common, schema_version="femx.mode/v2")


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (
            lambda: YeeGrid(  # type: ignore[arg-type]
                (FakeArray((2,)), FakeArray((2,))),
                SHA,
            ),
            "exactly three",
        ),
        (
            lambda: YeeGrid(
                (FakeArray((1,)), FakeArray((2,)), FakeArray((2,))),
                SHA,
            ),
            "at least two",
        ),
        (
            lambda: YeeGrid(
                (FakeArray((2,)), FakeArray((2,)), FakeArray((2,))),
                "bad",
            ),
            "coordinate digest",
        ),
        (
            lambda: YeeGrid(
                (FakeArray((2,)), FakeArray((2,)), FakeArray((2,))),
                SHA,
                coordinate_unit="um",
            ),
            "SI metres",
        ),
        (
            lambda: YeeVectorField(
                FakeArray((3, 1, 1, 2), dtype_kind="c"),
                yee_grid(shape=(1, 1, 1)),
                YeeFieldKind.ELECTRIC,
                "V/m",
            ),
            "must have shape",
        ),
        (
            lambda: YeeVectorField(
                FakeArray((3, 1, 1, 1), dtype_kind="c"),
                yee_grid(shape=(1, 1, 1)),
                YeeFieldKind.ELECTRIC,
                "V/m",
                representation=FieldRepresentation.CARTESIAN_SAMPLES,
            ),
            "Cartesian Yee",
        ),
        (
            lambda: YeeVectorField(
                FakeArray((3, 1, 1, 1), dtype_kind="c"),
                yee_grid(shape=(1, 1, 1)),
                YeeFieldKind.ELECTRIC,
                "",
            ),
            "declare unit",
        ),
        (
            lambda: YeeVectorField(
                FakeArray((3, 1, 1, 1), dtype_kind="f"),
                yee_grid(shape=(1, 1, 1)),
                YeeFieldKind.ELECTRIC,
                "V/m",
            ),
            "complex dtype",
        ),
    ],
)
def test_yee_contract_rejects_ambiguous_metadata(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"relative_power_error": math.nan}, "power error"),
        ({"relative_interpolation_error": math.inf}, "interpolation error"),
        ({"source_power_watts": 0.0}, "source power"),
        ({"pre_correction_power_watts": math.nan}, "pre-correction power"),
        ({"transferred_power_watts": -1.0}, "transferred power"),
        ({"power_correction_scale": math.inf}, "correction scale"),
        ({"relative_pre_correction_power_error": -1.0}, "pre-correction power error"),
        ({"target_runtime_version": None}, "runtime identity"),
        ({"target_source_revision": "bad"}, "source revision"),
        ({"target_source_digest": "bad"}, "source digest"),
    ],
)
def test_transfer_report_rejects_incomplete_or_nonfinite_evidence(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        replace(transfer_report(), **changes)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frequency_hz": math.nan}, "finite and positive"),
        ({"effective_index": complex(math.inf, 0.0)}, "must be finite"),
        ({"beta_per_m": complex(1.0, math.nan)}, "must be finite"),
        (
            {
                "electric": SampledVectorField(
                    FakeArray((3, 1), dtype_kind="c"),
                    (FakeArray((1,)),),
                    FieldRepresentation.CARTESIAN_SAMPLES,
                    "V/m",
                    "samples",
                )
            },
            "exact staggered Yee",
        ),
        (
            {"electric": yee_field(YeeFieldKind.MAGNETIC, "V/m")},
            "electric Yee offsets",
        ),
        (
            {"magnetic": yee_field(YeeFieldKind.ELECTRIC, "V/m")},
            "magnetic Yee offsets",
        ),
        (
            {
                "propagation": AxisDirection(Axis.X),
                "electric": yee_field(YeeFieldKind.ELECTRIC, "V/m"),
                "magnetic": yee_field(YeeFieldKind.MAGNETIC, "V/m"),
            },
            "one cell thick",
        ),
        (
            {
                "electric": YeeVectorField(
                    np.zeros((3, 4, 5, 1), dtype=np.complex64),
                    yee_grid(),
                    YeeFieldKind.ELECTRIC,
                    "V/m",
                ),
                "magnetic": YeeVectorField(
                    np.zeros((3, 4, 5, 1), dtype=np.complex128),
                    yee_grid(),
                    YeeFieldKind.MAGNETIC,
                    "V/m",
                ),
            },
            "different scalar dtypes",
        ),
        (
            {"electric": yee_field(YeeFieldKind.ELECTRIC, "A/m")},
            "electric Yee field",
        ),
        (
            {
                "transfer": replace(
                    transfer_report(),
                    source_representation=FieldRepresentation.CARTESIAN_SAMPLES,
                )
            },
            "originate from FEM",
        ),
        (
            {
                "transfer": replace(
                    transfer_report(),
                    target_representation=FieldRepresentation.CARTESIAN_SAMPLES,
                )
            },
            "target Cartesian Yee",
        ),
        (
            {"transfer": replace(transfer_report(), target_runtime_name="other")},
            "identify the FDTDX",
        ),
        (
            {"transfer": replace(transfer_report(), source_power_watts=None)},
            "complete signed-power",
        ),
        (
            {"normalization": ModeNormalization(target_power_watts=2.0)},
            "source power differs",
        ),
        (
            {"transfer": replace(transfer_report(), relative_power_error=0.1)},
            "transferred-power error",
        ),
        (
            {
                "transfer": replace(
                    transfer_report(),
                    relative_pre_correction_power_error=0.2,
                )
            },
            "pre-correction power error",
        ),
        (
            {"transfer": replace(transfer_report(), power_correction_scale=1.0)},
            "power-correction scale",
        ),
    ],
)
def test_mode_bundle_rejects_inconsistent_yee_and_power_contracts(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        ModeBundle(**{**mode_bundle_kwargs(), **changes})  # type: ignore[arg-type]
