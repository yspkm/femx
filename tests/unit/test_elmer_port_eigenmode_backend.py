from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support import structured_unit_square_mesh

from femx.backends.elmer.port_eigenmode import (
    ElmerPortEigenmodeBackend,
    ElmerPortEigenmodeIdentity,
    PreparedElmerPortEigenmode,
)
from femx.backends.elmer.runner import (
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)
from femx.backends.protocol import (
    BackendDescriptor,
    ExecutionPolicy,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.errors import (
    BackendError,
    BackendUnavailableError,
    ContractError,
    ExecutionNotAuthorizedError,
)
from femx.core.parameters import (
    ParameterReference,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import Problem
from femx.mesh import OrientationMap
from femx.physics import (
    PORT_LONGITUDINAL_POTENTIAL_FIELD,
    PORT_LONGITUDINAL_POTENTIAL_UNIT,
    PORT_TRANSVERSE_ELECTRIC_DOF_UNIT,
    PORT_TRANSVERSE_ELECTRIC_FIELD,
    VACUUM_SPEED_OF_LIGHT_M_PER_S,
    IsotropicOpticalRegion,
    PerfectElectricBoundary,
    PortEigenmode,
)
from femx.runtime import prepare, solve

pytestmark = pytest.mark.unit

EXPECTED_VERSION = "26.2-devel"
EXPECTED_REVISION = "abc123"
SOURCE_COMMIT = "a" * 40
SOURCE_DIGEST = "b" * 64


def _problem() -> Problem:
    mesh = structured_unit_square_mesh(1)
    cells = np.asarray(mesh.topology.connectivity)
    local_edges = cells[:, ((0, 1), (1, 2), (2, 0))]
    signs = np.where(local_edges[:, :, 0] < local_edges[:, :, 1], 1, -1).astype(np.int8)
    mesh = replace(mesh, orientation=OrientationMap(edge_signs=signs))
    physics = PortEigenmode(
        regions=(IsotropicOpticalRegion("domain", 12.0),),
        perfect_electric_boundaries=tuple(
            PerfectElectricBoundary(name) for name in ("bottom", "right", "top", "left")
        ),
        frequency_hz=VACUUM_SPEED_OF_LIGHT_M_PER_S / 1.55e-6,
        eigenmode_count=2,
        selected_mode_index=0,
        target_power_w=1.0,
    )
    return Problem("elmer-port", mesh, physics)


def _parameterized_problem() -> Problem:
    problem = _problem()
    physics = replace(
        problem.physics,
        regions=(IsotropicOpticalRegion("domain", ParameterReference("epsilon_r")),),
    )
    return Problem(
        "elmer-parameterized-port",
        problem.mesh,
        physics,
        parameters=ParameterSchema((ParameterSpec("epsilon_r", unit="1"),)),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_installation(tmp_path: Path) -> tuple[ElmerInstallation, ElmerPortEigenmodeIdentity]:
    root = tmp_path / "fake-elmer-install"
    executable = root / "bin" / "ElmerSolver"
    modules = root / "share" / "elmersolver" / "lib"
    executable.parent.mkdir(parents=True, exist_ok=True)
    modules.mkdir(parents=True, exist_ok=True)
    files = {
        executable: b"locked fake ElmerSolver\n",
        modules / "EMPort.so": b"locked fake EMPort\n",
        modules / "ResultOutputSolve.so": b"locked fake ResultOutput\n",
        modules / "SaveData.so": b"locked fake SaveData\n",
    }
    for path, content in files.items():
        if not path.exists():
            path.write_bytes(content)
    identity = ElmerPortEigenmodeIdentity(
        version=EXPECTED_VERSION,
        revision=EXPECTED_REVISION,
        executable_sha256=_sha256(executable),
        em_port_sha256=_sha256(modules / "EMPort.so"),
        result_output_sha256=_sha256(modules / "ResultOutputSolve.so"),
        save_data_sha256=_sha256(modules / "SaveData.so"),
        source_commit=SOURCE_COMMIT,
        source_digest=SOURCE_DIGEST,
    )
    return ElmerInstallation(executable.resolve()), identity


def _backend(tmp_path: Path, **overrides: object) -> ElmerPortEigenmodeBackend:
    installation, default = _fake_installation(tmp_path)
    identity = ElmerPortEigenmodeIdentity(
        version=overrides.pop("expected_version", default.version),  # type: ignore[arg-type]
        revision=overrides.pop("expected_revision", default.revision),  # type: ignore[arg-type]
        executable_sha256=overrides.pop(  # type: ignore[arg-type]
            "expected_executable_sha256", default.executable_sha256
        ),
        em_port_sha256=overrides.pop(  # type: ignore[arg-type]
            "expected_em_port_sha256", default.em_port_sha256
        ),
        result_output_sha256=overrides.pop(  # type: ignore[arg-type]
            "expected_result_output_sha256", default.result_output_sha256
        ),
        save_data_sha256=overrides.pop(  # type: ignore[arg-type]
            "expected_save_data_sha256", default.save_data_sha256
        ),
        source_commit=overrides.pop("source_commit", default.source_commit),  # type: ignore[arg-type]
        source_digest=overrides.pop("source_digest", default.source_digest),  # type: ignore[arg-type]
        source_worktree_state=overrides.pop(  # type: ignore[arg-type]
            "source_worktree_state", default.source_worktree_state
        ),
    )
    return ElmerPortEigenmodeBackend(installation, identity, **overrides)  # type: ignore[arg-type]


_COMPONENTS = (
    "ef2d re 1",
    "ef2d re 2",
    "ef2d re 3",
    "ef2d im 1",
    "ef2d im 2",
    "ef2d im 3",
)


def _field_record(
    save_count: int,
    *,
    mode_count: int = 2,
    constrained_raw: bool = False,
) -> str:
    timestep = save_count if save_count <= mode_count else 1
    lines = [f"Time: {save_count} {timestep} 1.00000000E+000", "eport re"]
    lines.extend(("Perm: 9 9", *(f"{index} {index}" for index in range(1, 10))))
    raw_real = np.zeros(9)
    raw_imaginary = np.zeros(9)
    if save_count <= mode_count:
        raw_real[6] = float(save_count)
        raw_imaginary[6] = -0.25 * save_count
        if constrained_raw:
            raw_real[0] = 1.0
    lines.extend(str(value) for value in raw_real)
    lines.extend(("eport im", "Perm: use previous"))
    lines.extend(str(value) for value in raw_imaginary)
    for component_index, name in enumerate(_COMPONENTS):
        lines.append(name)
        if component_index == 0:
            lines.extend(("Perm: 9 4", "1 4", "2 3", "3 2", "4 1"))
        else:
            lines.append("Perm: use previous")
        start = 4 * component_index + 1
        lines.extend(str(float(start + offset)) for offset in range(4))
    return "\n".join(lines)


def _result_text(*, record_count: int = 3, constrained_raw: bool = False) -> str:
    declarations = [
        "eport[eport re:1 eport im:1] : 18 9 2 : port mode",
        "ef2d[ef2d re:3 ef2d im:3] : 24 9 6 : port mode_post",
        "eport re : 9 9 1 : port mode",
        "eport im : 9 9 1 : port mode",
        *(f"{name} : 4 9 1 : port mode_post" for name in _COMPONENTS),
    ]
    return "\n".join(
        (
            "ASCII 3",
            "!dynamic timestamp",
            "Degrees of freedom:",
            *declarations,
            "Total DOFs: 8",
            "Number Of Nodes: 4",
            *(
                _field_record(index, constrained_raw=constrained_raw)
                for index in range(1, record_count + 1)
            ),
            "",
        )
    )


def _stdout(
    *,
    version: str = EXPECTED_VERSION,
    revision: str = EXPECTED_REVISION,
    tolerance: str = "1.000E-10",
    converged_count: int = 2,
    completed: bool = True,
    forward: bool = True,
) -> str:
    if forward:
        eigenvalue = "-1.6000000000000000E+01"
        beta_real, beta_imag = "4.000000E+00", "0.000000E+00"
        scalar_beta = "4.000000000000E+00"
    else:
        eigenvalue = "1.6000000000000000E+01"
        beta_real, beta_imag = "0.000000E+00", "-4.000000E+00"
        scalar_beta = "0.000000000000E+00"
    lines = [
        f"MAIN: Version: {version} (Rev: {revision}, Compiled: test)",
        f"EigenSolveComplex: Convergence criterion is: {tolerance}",
        "EigenSolveComplex: Number of eigensystem iterations is: 17",
        f"EigenSolveComplex: Number of converged Ritz values is: {converged_count}",
        f"EigenSolveComplex: 1 ( {eigenvalue}, 0.0 )",
        "EigenSolveComplex: 2 ( -9.0000000000000000E+00, 0.0 )",
        "CheckResidualsComplex: L^2 Norm of the residual: 1 1.0E-08",
        "CheckResidualsComplex: L^2 Norm of the residual: 2 2.0E-08",
        f"EMPortSolver: Propagation constant beta: {beta_real} {beta_imag}",
        f"SaveScalars: 1: res: port beta 1 {scalar_beta}",
        "EMPortSolver: Port power: 4.000000E+00",
        "SaveScalars: 2: res: port power 1 4.000000000000E+00",
        "SaveScalars: 3: res: port impedance 1 5.000000000000E-01",
    ]
    if completed:
        lines.append("MAIN: *** Elmer Solver: ALL DONE ***")
    return "\n".join(lines) + "\n"


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_code: int = 0,
    stdout: str | None = None,
    write_result: bool = True,
    write_vtu: bool = True,
    record_count: int = 3,
    constrained_raw: bool = False,
    mutate_after: str | None = None,
) -> None:
    def fake_run(self, command, *, working_directory, policy):
        modules = self.installation.executable.resolve().parent.parent / "share/elmersolver/lib"
        assert command.arguments == ()
        assert command.environment == {
            "ELMER_HOME": str(self.installation.executable.resolve().parent.parent),
            "ELMER_LIB": str(modules),
            "ELMER_MODULES_PATH": str(modules),
        }
        policy.require_external_process(component_name="elmer-port-eigenmode")
        if write_result:
            (working_directory / "mesh/femx.result").write_text(
                _result_text(
                    record_count=record_count,
                    constrained_raw=constrained_raw,
                ),
                encoding="utf-8",
            )
        if write_vtu:
            (working_directory / "mesh/femx-mode_t0001.vtu").write_bytes(b"raw-vtu")
        if mutate_after is not None:
            (modules / mutate_after).write_bytes(b"changed during execution\n")
        return ElmerProcessResult(
            argv=(str(self.installation.executable),),
            return_code=return_code,
            stdout=_stdout() if stdout is None else stdout,
            stderr="fake-stderr",
            elapsed_seconds=0.25,
        )

    monkeypatch.setattr(ElmerRunner, "run", fake_run)


def _authorized(run_directory: Path, *, parameters: ParameterValues | None = None) -> SolveRequest:
    return SolveRequest(
        parameters=ParameterValues() if parameters is None else parameters,
        run_directory=run_directory,
        policy=ExecutionPolicy(execution_authorized=True, allow_external_process=True),
    )


def test_port_backend_prepare_is_pure_and_success_retains_full_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_run(monkeypatch)
    run_directory = tmp_path / "attempt-001"
    backend = _backend(tmp_path)

    prepared = prepare(
        _problem(),
        backend,
        request=PrepareRequest(run_directory=run_directory),
    )
    assert isinstance(prepared.payload, PreparedElmerPortEigenmode)
    assert not run_directory.exists()
    assert 'Procedure = File "' in prepared.payload.sif

    solution = solve(prepared, backend, request=_authorized(run_directory))

    assert solution.convergence.status.value == "converged"
    assert solution.convergence.iterations == 17
    assert solution.convergence.residual_norm is None
    assert solution.convergence.tolerance == 1.0e-10
    assert solution.fields["electric_field"].values.shape == (4, 3)
    assert solution.fields["electric_field"].values.dtype == np.complex128
    assert solution.fields["electric_field"].unit == "V/m"
    assert solution.fields["electric_field"].function_space.value_shape == (3,)
    scalar_coefficients = solution.fields[PORT_LONGITUDINAL_POTENTIAL_FIELD]
    edge_coefficients = solution.fields[PORT_TRANSVERSE_ELECTRIC_FIELD]
    assert scalar_coefficients.values.shape == (4,)
    assert edge_coefficients.values.shape == (5,)
    assert scalar_coefficients.values.dtype == edge_coefficients.values.dtype == np.complex128
    assert scalar_coefficients.unit == PORT_LONGITUDINAL_POTENTIAL_UNIT
    assert edge_coefficients.unit == PORT_TRANSVERSE_ELECTRIC_DOF_UNIT
    assert solution.observables["propagation_constant_rad_per_m"] == 4.0 + 0.0j
    assert solution.observables["selected_eigenvalue_per_m2"] == -16.0 + 0.0j
    assert solution.observables["raw_forward_power_W"] == 4.0
    assert solution.observables["target_forward_power_W"] == 1.0
    assert solution.observables["field_amplitude_scale"] == 0.5
    assert solution.observables["maximum_requested_eigen_residual"] == 2.0e-8
    assert solution.metadata["projected_field_record_count"] == "3"
    assert solution.metadata["raw_mixed_mode_count"] == "2"
    assert solution.metadata["raw_mixed_edge_count"] == "5"
    assert solution.metadata["raw_mixed_final_zero_verified"] == "true"
    assert solution.metadata["raw_mixed_pec_zero_verified"] == "true"
    assert solution.metadata["fdtdx_mode_bundle_status"].startswith("requires_explicit_hashed")
    assert solution.metadata["printed_residual_semantics"].startswith("scale_dependent")
    assert solution.metadata["elmer_source_commit"] == SOURCE_COMMIT
    for key in (
        "elmer_executable_sha256",
        "elmer_em_port_sha256",
        "elmer_result_output_sha256",
        "elmer_save_data_sha256",
        "input_sif_sha256",
        "result_sha256",
        "raw_vtu_sha256",
        "spectrum_sha256",
    ):
        assert len(solution.metadata[key]) == 64
    spectrum = json.loads((run_directory / "port-spectrum.json").read_text(encoding="utf-8"))
    assert spectrum["schema_version"] == "femx.elmer-port-spectrum/v1"
    assert spectrum["eigenvalues_per_m2"][0] == {"real": -16.0, "imag": 0.0}
    assert (run_directory / "mesh/femx-mode_t0001.vtu").read_bytes() == b"raw-vtu"
    assert (run_directory / "elmer.stdout.log").read_text(encoding="utf-8") == _stdout()


def test_port_backend_requires_dual_authorization_before_writing(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    run_directory = tmp_path / "denied"
    prepared = backend.prepare(_problem(), PrepareRequest(run_directory=run_directory))

    with pytest.raises(ExecutionNotAuthorizedError, match="requires"):
        backend.solve(prepared, SolveRequest(run_directory=run_directory))
    assert not run_directory.exists()


@pytest.mark.parametrize(
    ("key", "message"),
    [
        ("expected_executable_sha256", "executable SHA-256"),
        ("expected_em_port_sha256", "EMPort SHA-256"),
        ("expected_result_output_sha256", "ResultOutputSolve SHA-256"),
        ("expected_save_data_sha256", "SaveData SHA-256"),
    ],
)
def test_port_backend_rejects_locked_installation_hash_mismatch(
    tmp_path: Path, key: str, message: str
) -> None:
    with pytest.raises(BackendUnavailableError, match=message):
        _backend(tmp_path, **{key: "0" * 64})


def test_port_backend_validates_constructor_identity_and_payload(tmp_path: Path) -> None:
    _, identity = _fake_installation(tmp_path)
    with pytest.raises(BackendUnavailableError, match="does not exist"):
        ElmerPortEigenmodeBackend(ElmerInstallation((tmp_path / "missing").resolve()), identity)
    with pytest.raises(ContractError, match="timeout"):
        _backend(tmp_path, timeout_seconds=0.0)
    with pytest.raises(ContractError, match="convergence"):
        _backend(tmp_path, convergence_tolerance=float("nan"))
    with pytest.raises(ContractError, match="version"):
        _backend(tmp_path, expected_version=" bad ")
    with pytest.raises(ContractError, match="revision"):
        _backend(tmp_path, expected_revision="bad\nrevision")
    with pytest.raises(ContractError, match="SHA-256"):
        _backend(tmp_path, expected_em_port_sha256="bad")
    with pytest.raises(ContractError, match="Git SHA-1"):
        _backend(tmp_path, source_commit="short")
    with pytest.raises(ContractError, match="worktree state"):
        _backend(tmp_path, source_worktree_state="unknown")

    backend = _backend(tmp_path)
    valid = backend.prepare(_problem(), PrepareRequest())
    wrong_descriptor = PreparedProblem(
        BackendDescriptor("other", "1"), valid.problem, valid.payload
    )
    with pytest.raises(BackendError, match="identity"):
        backend.solve(wrong_descriptor, _authorized(tmp_path / "wrong-id"))
    wrong_payload = PreparedProblem(backend.descriptor, valid.problem, object())
    with pytest.raises(BackendError, match="payload"):
        backend.solve(wrong_payload, _authorized(tmp_path / "wrong-payload"))
    with pytest.raises(ContractError, match="parameter keys"):
        backend.solve(
            valid,
            _authorized(tmp_path / "parameters", parameters=ParameterValues({"x": 1.0})),
        )


def test_port_backend_renders_parameterized_material_at_solve_time(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_run(monkeypatch)
    backend = _backend(tmp_path)
    prepared = backend.prepare(_parameterized_problem(), PrepareRequest())
    assert isinstance(prepared.payload, PreparedElmerPortEigenmode)
    assert prepared.payload.sif is None

    run_directory = tmp_path / "parameterized-attempt"
    solution = backend.solve(
        prepared,
        _authorized(
            run_directory,
            parameters=ParameterValues({"epsilon_r": 11.75}),
        ),
    )

    sif = (run_directory / "case.sif").read_text(encoding="utf-8")
    assert "Relative Permittivity = Real 1.17500000000000000e+01" in sif
    assert solution.convergence.status.value == "converged"


@pytest.mark.parametrize(
    ("relative_path", "message"),
    [
        (Path("bin/ElmerSolver"), "executable SHA-256"),
        (Path("share/elmersolver/lib/EMPort.so"), "EMPort SHA-256"),
        (Path("share/elmersolver/lib/ResultOutputSolve.so"), "ResultOutputSolve SHA-256"),
        (Path("share/elmersolver/lib/SaveData.so"), "SaveData SHA-256"),
    ],
)
def test_port_backend_reverifies_every_runtime_file_before_execution(
    tmp_path: Path, relative_path: Path, message: str
) -> None:
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    (tmp_path / "fake-elmer-install" / relative_path).write_bytes(b"changed after prepare\n")

    with pytest.raises(BackendUnavailableError, match=message):
        backend.solve(prepared, _authorized(tmp_path / "identity-mismatch"))
    assert not (tmp_path / "identity-mismatch").exists()


def test_port_backend_rejects_installation_change_during_execution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_run(monkeypatch, mutate_after="SaveData.so")
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())

    with pytest.raises(BackendUnavailableError, match="SaveData SHA-256"):
        backend.solve(prepared, _authorized(tmp_path / "changed-during-run"))


def test_port_backend_requires_one_fresh_absolute_attempt_directory(tmp_path: Path) -> None:
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    policy = ExecutionPolicy(execution_authorized=True, allow_external_process=True)
    with pytest.raises(ContractError, match="explicit durable"):
        backend.solve(prepared, SolveRequest(policy=policy))

    different = backend.prepare(_problem(), PrepareRequest(run_directory=tmp_path / "prepared"))
    with pytest.raises(ContractError, match="different"):
        backend.solve(different, _authorized(tmp_path / "solved"))

    with pytest.raises(ContractError, match="absolute"):
        backend.solve(prepared, _authorized(Path("relative-attempt")))

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("owned", encoding="utf-8")
    with pytest.raises(BackendError, match="empty"):
        backend.solve(prepared, _authorized(occupied))


def test_port_backend_accepts_existing_empty_attempt_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_run(monkeypatch)
    run_directory = tmp_path / "empty"
    run_directory.mkdir()
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())

    solution = backend.solve(prepared, _authorized(run_directory))

    assert solution.convergence.status.value == "converged"


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"return_code": 3}, "return code 3"),
        ({"stdout": _stdout(completed=False)}, "completion marker"),
        ({"stdout": "MAIN: *** Elmer Solver: ALL DONE ***\n"}, "version and revision"),
        ({"stdout": _stdout(version="wrong", revision="wrong")}, "differs from the locked"),
        ({"write_result": False}, "does not exist"),
        ({"write_vtu": False}, "raw VTU"),
        ({"record_count": 2}, "save/timestep sequence"),
        ({"stdout": _stdout(tolerance="1.000E-09")}, "different eigen convergence"),
        ({"stdout": _stdout(forward=False)}, "non-forward"),
    ],
)
def test_port_backend_fails_closed_on_process_identity_numerics_and_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    options: dict[str, object],
    message: str,
) -> None:
    _install_fake_run(monkeypatch, **options)  # type: ignore[arg-type]
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())
    run_directory = tmp_path / message.replace(" ", "-").replace("/", "-")

    with pytest.raises(BackendError, match=message):
        backend.solve(prepared, _authorized(run_directory))
    assert (run_directory / "case.sif").is_file()
    assert (run_directory / "elmer.stdout.log").is_file()


def test_port_backend_reports_incomplete_ritz_convergence_without_hiding_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install_fake_run(monkeypatch, stdout=_stdout(converged_count=1))
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())

    solution = backend.solve(prepared, _authorized(tmp_path / "not-converged"))

    assert solution.convergence.status.value == "not_converged"
    assert solution.fields["electric_field"].values.shape == (4, 3)


def test_port_backend_rejects_nonzero_raw_pec_coefficients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_fake_run(monkeypatch, constrained_raw=True)
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())

    with pytest.raises(BackendError, match="PEC-constrained"):
        backend.solve(prepared, _authorized(tmp_path / "nonzero-pec"))


def test_port_backend_rejects_a_partial_computed_spectrum(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    partial = "\n".join(
        line
        for line in _stdout(converged_count=1).splitlines()
        if not line.startswith("EigenSolveComplex: 2 (")
        and not line.startswith("CheckResidualsComplex: L^2 Norm of the residual: 2 ")
    )
    _install_fake_run(monkeypatch, stdout=partial + "\n")
    backend = _backend(tmp_path)
    prepared = backend.prepare(_problem(), PrepareRequest())

    with pytest.raises(BackendError, match="every requested eigenpair"):
        backend.solve(prepared, _authorized(tmp_path / "partial-spectrum"))
