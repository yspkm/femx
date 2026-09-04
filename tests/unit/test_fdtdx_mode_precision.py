from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.errors import ContractError
from femx.interop.fdtdx import (
    FDTDXFingerprint,
    FieldRepresentation,
    MagneticFieldConvention,
    ModeBundle,
    ModeNormalization,
    SolverFingerprint,
    TransferReport,
    YeeFieldKind,
    YeeVectorField,
    build_fdtdx_mode_source_contract,
    build_yee_grid,
    lower_mode_source_inputs_for_tpu,
)
from femx.interop.fdtdx import mode_precision as precision_module
from femx.physics import VACUUM_SPEED_OF_LIGHT_M_PER_S

pytestmark = pytest.mark.unit

_SHA = "a" * 64
_FDTDX = FDTDXFingerprint(
    package_version="0.6.2",
    source_revision="81a58da9cde4a4ff822f835b63597c0d0d8ba978",
    source_digest="c881f0cf32b4272dc10e0acee716bfda3cf2d0f11973c23d31f29b11f3dce01c",
)


def _bundle() -> ModeBundle:
    x_edges = np.asarray((-300e-9, -100e-9, 100e-9, 300e-9), dtype=np.float64)
    y_edges = np.asarray((-240e-9, 0.0, 240e-9), dtype=np.float64)
    z_edges = np.asarray((80e-9, 120e-9), dtype=np.float64)
    grid = build_yee_grid((x_edges, y_edges, z_edges))
    area = float(np.ptp(x_edges) * np.ptp(y_edges))
    effective_index = math.sqrt(2.085136)
    eta0 = 4.0e-7 * math.pi * VACUUM_SPEED_OF_LIGHT_M_PER_S
    amplitude = math.sqrt(2.0 * eta0 / (effective_index * area))
    electric = np.zeros((3, *grid.shape), dtype=np.complex128)
    magnetic = np.zeros_like(electric)
    electric[0] = amplitude * np.exp(0.125j)
    magnetic[1] = effective_index * electric[0]
    frequency_hz = VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6
    return ModeBundle(
        frequency_hz=frequency_hz,
        effective_index=effective_index + 0.0j,
        beta_per_m=effective_index * 2.0 * math.pi * frequency_hz / VACUUM_SPEED_OF_LIGHT_M_PER_S,
        electric=YeeVectorField(electric, grid, YeeFieldKind.ELECTRIC, "V/m"),
        magnetic=YeeVectorField(magnetic, grid, YeeFieldKind.MAGNETIC, "V/m"),
        propagation=AxisDirection(Axis.Z, Direction.POSITIVE),
        magnetic_convention=MagneticFieldConvention.ETA0_H,
        normalization=ModeNormalization(target_power_watts=1.0),
        solver=SolverFingerprint("test-port", "1", _SHA, _SHA, "revision"),
        transfer=TransferReport(
            source_representation=FieldRepresentation.FEM_DOFS,
            target_representation=FieldRepresentation.CARTESIAN_YEE_SAMPLES,
            operator_sha256="b" * 64,
            relative_power_error=0.0,
            source_power_watts=1.0,
            pre_correction_power_watts=1.0,
            relative_pre_correction_power_error=0.0,
            transferred_power_watts=1.0,
            power_correction_scale=1.0,
            target_runtime_name="fdtdx",
            target_runtime_version=_FDTDX.package_version,
            target_source_revision=_FDTDX.source_revision,
            target_source_digest=_FDTDX.source_digest,
        ),
    )


def _medium(bundle: ModeBundle) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full((1, *bundle.electric.grid.shape), 1.0 / 2.085136, dtype=np.float64),
        np.asarray(1.0, dtype=np.float64),
    )


def _lower(bundle: ModeBundle | None = None):
    bundle = _bundle() if bundle is None else bundle
    inverse_permittivity, inverse_permeability = _medium(bundle)
    return lower_mode_source_inputs_for_tpu(
        bundle,
        expected_inverse_permittivity=inverse_permittivity,
        expected_inverse_permeability=inverse_permeability,
    )


def test_tpu_precision_lowering_is_explicit_deterministic_and_power_preserving() -> None:
    source = _bundle()
    source_electric = np.array(source.electric.values, copy=True)
    first = _lower(source)
    second = _lower(source)

    assert str(first.bundle.electric.values.dtype) == "complex64"
    assert str(first.bundle.magnetic.values.dtype) == "complex64"
    assert all(
        np.asarray(axis).dtype == np.dtype("<f4")
        for axis in first.bundle.electric.grid.edge_coordinates
    )
    assert np.asarray(first.expected_inverse_permittivity).dtype == np.dtype("<f4")
    assert np.asarray(first.expected_inverse_permeability).dtype == np.dtype("<f8")
    source_inverse_permittivity, _source_inverse_permeability = _medium(source)
    expected_runtime_inverse = np.float32(1.0) / np.asarray(
        np.reciprocal(source_inverse_permittivity),
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        first.expected_inverse_permittivity,
        expected_runtime_inverse,
    )
    assert not np.asarray(first.bundle.electric.values).flags.writeable
    assert not np.asarray(first.expected_inverse_permittivity).flags.writeable
    np.testing.assert_array_equal(source.electric.values, source_electric)

    report = first.report
    assert report.schema_version == "femx.fdtdx.mode_precision_lowering/v1"
    assert report.source_field_dtype == "complex128"
    assert report.runtime_real_dtype == "float32"
    assert report.runtime_field_dtype == "complex64"
    assert report.target_backend == "tpu"
    assert report.precision_fallback is False
    assert report.maximum_coordinate_error_cell_fraction < 2.0e-7
    assert report.relative_frequency_error < 1.0e-7
    assert report.relative_effective_index_error < 1.0e-7
    assert report.relative_beta_error < 1.0e-7
    assert report.electric_relative_l2_error < 2.0e-7
    assert report.magnetic_relative_l2_error < 2.0e-7
    assert report.inverse_permittivity_maximum_relative_error < 1.0e-7
    assert report.relative_pre_correction_power_error < 2.0e-7
    assert report.relative_power_error < 2.0e-7
    assert report.transferred_power_watts == pytest.approx(1.0, rel=2.0e-7)
    assert len(report.source_bundle_sha256) == 64
    assert len(report.runtime_bundle_sha256) == 64
    assert len(report.lowering_operator_sha256) == 64
    assert len(report.sha256) == 64
    assert report.canonical_data() == second.report.canonical_data()
    assert report.sha256 == second.report.sha256
    assert "not TPU execution" in str(report.canonical_data()["claim_scope"])


def test_lowered_values_satisfy_the_existing_static_source_contract() -> None:
    lowered = _lower()
    contract = build_fdtdx_mode_source_contract(
        lowered.bundle,
        source_name="fem-port",
        expected_inverse_permittivity=lowered.expected_inverse_permittivity,
        expected_inverse_permeability=lowered.expected_inverse_permeability,
        fdtdx=_FDTDX,
    )

    assert contract.field_dtype == "complex64"
    assert np.asarray(contract.expected_inverse_permittivity).dtype == np.dtype("<f4")
    assert np.asarray(contract.expected_inverse_permeability).dtype == np.dtype("<f8")
    assert contract.mode_bundle_sha256 == lowered.report.runtime_bundle_sha256


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_bundle_sha256", "bad", "source bundle digest"),
        ("lowering_operator_sha256", "bad", "lowering operator digest"),
        ("source_field_dtype", "complex256", "source field dtype"),
        ("relative_frequency_error", math.inf, "frequency error"),
        ("pre_correction_power_watts", 0.0, "pre-correction power"),
        ("runtime_real_dtype", "float64", "explicit float32"),
        ("precision_fallback", True, "explicit float32"),
        ("schema_version", "future", "unsupported"),
        ("relative_beta_error", 2.0e-6, "exceeds"),
    ],
)
def test_precision_report_rejects_invalid_claims(field: str, value: object, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        replace(_lower().report, **{field: value})


def test_lowering_rejects_incompatible_mode_semantics_and_collapsed_coordinates() -> None:
    bundle = _bundle()
    with pytest.raises(ContractError, match="positive-z"):
        _lower(replace(bundle, propagation=AxisDirection(Axis.Z, Direction.NEGATIVE)))

    physical_magnetic = replace(bundle.magnetic, unit="A/m")
    with pytest.raises(ContractError, match="eta0_H"):
        _lower(
            replace(
                bundle,
                magnetic=physical_magnetic,
                magnetic_convention=MagneticFieldConvention.PHYSICAL_H,
            )
        )

    with pytest.raises(ContractError, match="positive forward optical scalars"):
        _lower(replace(bundle, beta_per_m=0.0j))

    collapsed_grid = build_yee_grid(
        (
            np.asarray((1.0, 1.0 + 1.0e-9, 1.0 + 2.0e-9, 1.0 + 3.0e-9)),
            bundle.electric.grid.edge_coordinates[1],
            bundle.electric.grid.edge_coordinates[2],
        )
    )
    collapsed = replace(
        bundle,
        electric=replace(bundle.electric, grid=collapsed_grid),
        magnetic=replace(bundle.magnetic, grid=collapsed_grid),
    )
    with pytest.raises(ContractError, match="collapse"):
        _lower(collapsed)


def test_lowering_rejects_invalid_or_backward_fields() -> None:
    bundle = _bundle()
    zero_magnetic = replace(
        bundle,
        magnetic=replace(bundle.magnetic, values=np.zeros_like(bundle.magnetic.values)),
    )
    with pytest.raises(ContractError, match="zero, non-finite, or backward"):
        _lower(zero_magnetic)

    backward = replace(
        bundle,
        magnetic=replace(bundle.magnetic, values=-np.asarray(bundle.magnetic.values)),
    )
    with pytest.raises(ContractError, match="zero, non-finite, or backward"):
        _lower(backward)


def test_nonuniform_complex64_power_is_reported_without_rewriting_fem_transfer() -> None:
    bundle = _bundle()
    rng = np.random.default_rng(7)
    electric = np.array(bundle.electric.values, copy=True)
    magnetic = np.array(bundle.magnetic.values, copy=True)
    electric[0] *= 1.0 + 0.01 * rng.normal(size=electric[0].shape)
    magnetic[1] *= 1.0 + 0.02 * rng.normal(size=magnetic[1].shape)
    perturbed_power = precision_module._signed_power_watts(
        electric,
        magnetic,
        bundle.electric.grid,
    )
    scale = math.sqrt(bundle.normalization.target_power_watts / perturbed_power)
    electric *= scale
    magnetic *= scale
    perturbed = replace(
        bundle,
        electric=replace(bundle.electric, values=electric),
        magnetic=replace(bundle.magnetic, values=magnetic),
    )

    lowered = _lower(perturbed)

    assert lowered.bundle.transfer == perturbed.transfer
    assert lowered.report.relative_power_error < 2.0e-7
    assert lowered.report.runtime_bundle_sha256 != lowered.report.source_bundle_sha256


@pytest.mark.parametrize(
    ("permittivity", "permeability", "message"),
    [
        (np.ones((1, 2, 2, 1)), np.asarray(1.0), "real shape"),
        (np.full((1, 3, 2, 1), np.nan), np.asarray(1.0), "finite and positive"),
        (np.ones((1, 3, 2, 1)), np.ones((1,)), "one real scalar"),
        (np.ones((1, 3, 2, 1)), np.asarray(0.5), "finite scalar one"),
    ],
)
def test_lowering_rejects_invalid_source_medium(
    permittivity: np.ndarray,
    permeability: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ContractError, match=message):
        lower_mode_source_inputs_for_tpu(
            _bundle(),
            expected_inverse_permittivity=permittivity,
            expected_inverse_permeability=permeability,
        )


def test_runtime_input_container_rejects_drift_from_report() -> None:
    lowered = _lower()
    with pytest.raises(ContractError, match="differs from its precision report"):
        replace(
            lowered,
            bundle=replace(lowered.bundle, beta_per_m=lowered.bundle.beta_per_m * 1.01),
        )


def test_precision_helpers_fail_closed_on_noncanonical_values() -> None:
    with pytest.raises(ContractError, match="canonical JSON"):
        precision_module._canonical_json({"invalid": math.nan})
    assert precision_module._relative_l2(np.zeros(2), np.zeros(2)) == 0.0
    assert math.isinf(precision_module._relative_l2(np.ones(2), np.zeros(2)))


def test_lowering_rejects_unsupported_or_nonfinite_source_fields() -> None:
    bundle = _bundle()
    extended = np.asarray(bundle.electric.values, dtype=np.clongdouble)
    unsupported = replace(
        bundle,
        electric=replace(bundle.electric, values=extended),
        magnetic=replace(
            bundle.magnetic, values=np.asarray(bundle.magnetic.values, dtype=np.clongdouble)
        ),
    )
    with pytest.raises(ContractError, match="complex64 or complex128"):
        _lower(unsupported)

    nonfinite_values = np.array(bundle.electric.values, copy=True)
    nonfinite_values[0, 0, 0, 0] = complex(math.nan, 0.0)
    nonfinite = replace(bundle, electric=replace(bundle.electric, values=nonfinite_values))
    with pytest.raises(ContractError, match="fields must be finite"):
        _lower(nonfinite)


def test_lowering_rejects_invalid_correction_and_corrected_power(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle()
    huge_target = 1.0e300
    huge = replace(
        bundle,
        normalization=replace(bundle.normalization, target_power_watts=huge_target),
        transfer=replace(
            bundle.transfer,
            source_power_watts=huge_target,
            pre_correction_power_watts=1.0,
            relative_pre_correction_power_error=1.0,
            transferred_power_watts=huge_target,
            power_correction_scale=1.0e150,
        ),
    )
    with pytest.warns(RuntimeWarning, match="overflow"):
        with pytest.raises(ContractError, match="invalid power correction"):
            _lower(huge)

    calls = iter((1.0, 0.0))
    monkeypatch.setattr(precision_module, "_signed_power_watts", lambda *_args: next(calls))
    with pytest.raises(ContractError, match="invalid corrected power"):
        _lower(bundle)


def test_runtime_input_container_rejects_runtime_scalar_drift() -> None:
    lowered = _lower()
    source = _bundle()
    complex128_bundle = replace(
        lowered.bundle,
        electric=replace(lowered.bundle.electric, values=source.electric.values),
        magnetic=replace(lowered.bundle.magnetic, values=source.magnetic.values),
    )
    with pytest.raises(ContractError, match="fields must use complex64"):
        replace(lowered, bundle=complex128_bundle)

    float64_grid_bundle = replace(
        lowered.bundle,
        electric=replace(lowered.bundle.electric, grid=source.electric.grid),
        magnetic=replace(lowered.bundle.magnetic, grid=source.magnetic.grid),
    )
    with pytest.raises(ContractError, match="coordinates must use float32"):
        replace(lowered, bundle=float64_grid_bundle)

    with pytest.raises(ContractError, match="permittivity has the wrong"):
        replace(
            lowered,
            expected_inverse_permittivity=np.asarray(
                lowered.expected_inverse_permittivity,
                dtype=np.float64,
            ),
        )
    with pytest.raises(ContractError, match="float64 scalar-one"):
        replace(
            lowered,
            expected_inverse_permeability=np.asarray(1.0, dtype=np.float32),
        )
    with pytest.raises(ContractError, match="arrays must be read-only"):
        replace(
            lowered,
            expected_inverse_permittivity=np.array(
                lowered.expected_inverse_permittivity,
                copy=True,
            ),
        )
