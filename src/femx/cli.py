"""Read-only command-line diagnostics for the femx harness."""

import argparse
import importlib.util
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from femx import __version__
from femx.backends.elmer import ElmerInstallation


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One side-effect-free environment diagnostic."""

    name: str
    available: bool
    detail: str
    required: bool = False


def collect_doctor_checks(*, require: frozenset[str] = frozenset()) -> tuple[DoctorCheck, ...]:
    """Inspect imports and executable paths without importing JAX or running Elmer."""

    python_supported = (3, 11) <= sys.version_info[:2] < (3, 15)
    checks: list[DoctorCheck] = [
        DoctorCheck(
            name="python",
            available=python_supported,
            detail=platform.python_version(),
            required=True,
        )
    ]
    for module_name in ("jax", "fdtdx", "h5py", "meshio"):
        available = importlib.util.find_spec(module_name) is not None
        checks.append(
            DoctorCheck(
                name=module_name,
                available=available,
                detail="installed" if available else "not installed",
                required=module_name in require,
            )
        )
    installation = ElmerInstallation.discover()
    checks.append(
        DoctorCheck(
            name="elmer",
            available=installation is not None,
            detail=str(installation.executable) if installation is not None else "not on PATH",
            required="elmer" in require,
        )
    )
    return tuple(checks)


def _doctor(args: argparse.Namespace) -> int:
    required = frozenset(args.require)
    checks = collect_doctor_checks(require=required)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2, sort_keys=True))
    else:
        for check in checks:
            status = "ok" if check.available else "missing"
            requirement = " required" if check.required else " optional"
            print(f"{check.name:8} {status:7} {requirement}: {check.detail}")
        print("No accelerator was initialized and no external process was executed.")
    return 1 if any(check.required and not check.available for check in checks) else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without probing the environment."""

    parser = argparse.ArgumentParser(prog="femx")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="inspect optional dependencies without execution")
    doctor.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    doctor.add_argument(
        "--require",
        action="append",
        choices=("jax", "fdtdx", "h5py", "meshio", "elmer"),
        default=[],
        help="treat one optional component as required; may be repeated",
    )
    doctor.set_defaults(handler=_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the femx command line."""

    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))
