#!/usr/bin/env python3
"""Validate the external Elmer and FDTDX development source checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, cast
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCK_PATH = REPOSITORY_ROOT / "harness" / "source-baselines.toml"
SCHEMA_VERSION = "femx.source-baselines/v1"
REPORT_SCHEMA_VERSION = "femx.source-checkouts/v1"

_GIT_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*$")


class SourceConfigurationError(ValueError):
    """A source-baseline lock is malformed."""


@dataclass(frozen=True, slots=True)
class GitCommandResult:
    """Captured result of one bounded, read-only Git command."""

    return_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class GitClient(Protocol):
    """Minimal Git command boundary used by the source inspector."""

    def run(
        self,
        repository: Path,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> GitCommandResult:
        """Run one read-only Git command."""


class SubprocessGitClient:
    """Shell-free implementation of the bounded Git command boundary."""

    def run(
        self,
        repository: Path,
        arguments: tuple[str, ...],
        *,
        timeout_seconds: float,
    ) -> GitCommandResult:
        """Run Git without a shell and capture text output."""

        try:
            completed = subprocess.run(
                ("git", "-C", str(repository), *arguments),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        except FileNotFoundError:
            return GitCommandResult(127, stderr="git executable not found")
        except subprocess.TimeoutExpired:
            return GitCommandResult(124, stderr="git command timed out", timed_out=True)
        return GitCommandResult(completed.returncode, completed.stdout, completed.stderr)


@dataclass(frozen=True, slots=True)
class SourceSpec:
    """Expected identity and source anchors for one development checkout."""

    name: str
    purpose: str
    checkout: Path
    checkout_reference: str
    baseline_commit: str
    remotes: Mapping[str, str]
    required_files: tuple[PurePosixPath, ...]
    required_directories: tuple[PurePosixPath, ...]

    def __post_init__(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.name):
            raise SourceConfigurationError(f"invalid source name: {self.name!r}")
        if not self.purpose or self.purpose.strip() != self.purpose:
            raise SourceConfigurationError(f"source {self.name!r} has an invalid purpose")
        if not self.checkout.is_absolute():
            raise SourceConfigurationError(f"source {self.name!r} checkout must be resolved")
        if not self.checkout_reference:
            raise SourceConfigurationError(f"source {self.name!r} checkout reference is empty")
        if not _GIT_COMMIT_PATTERN.fullmatch(self.baseline_commit):
            raise SourceConfigurationError(
                f"source {self.name!r} baseline must be a lowercase 40-character Git commit"
            )
        if not self.remotes:
            raise SourceConfigurationError(f"source {self.name!r} must declare a remote")
        normalized_remotes: dict[str, str] = {}
        for remote, repository in self.remotes.items():
            if not remote or remote.strip() != remote:
                raise SourceConfigurationError(f"source {self.name!r} has an invalid remote name")
            normalized = normalize_repository_url(repository)
            if not normalized:
                raise SourceConfigurationError(f"source {self.name!r} remote {remote!r} is invalid")
            normalized_remotes[remote] = normalized
        object.__setattr__(self, "remotes", MappingProxyType(normalized_remotes))
        _validate_anchor_paths(self.name, self.required_files, kind="file")
        _validate_anchor_paths(self.name, self.required_directories, kind="directory")


@dataclass(frozen=True, slots=True)
class SourceCheckoutReport:
    """Machine-readable identity and availability report for one checkout."""

    name: str
    purpose: str
    checkout_reference: str
    baseline_commit: str
    head_commit: str | None
    branch: str | None
    remotes: Mapping[str, str]
    missing_files: tuple[str, ...]
    missing_directories: tuple[str, ...]
    worktree_state: str
    worktree_entry_count: int | None
    worktree_status_digest: str | None
    errors: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether identity and anchor requirements passed."""

        return not self.errors

    @property
    def baseline_matches(self) -> bool:
        """Whether the checkout is at the locked commit."""

        return self.head_commit == self.baseline_commit

    @property
    def reproducibility_state(self) -> str:
        """Describe what this inspection establishes about source reproducibility."""

        if not self.valid:
            return "invalid"
        if self.worktree_state == "clean":
            return "clean"
        if self.worktree_state == "dirty":
            return "dirty"
        return "identity_only"

    @property
    def source_digest(self) -> str:
        """Hash the source identity without embedding an absolute local path."""

        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "name": self.name,
            "baseline_commit": self.baseline_commit,
            "head_commit": self.head_commit,
            "branch": self.branch,
            "remotes": dict(sorted(self.remotes.items())),
            "missing_files": list(self.missing_files),
            "missing_directories": list(self.missing_directories),
            "worktree_state": self.worktree_state,
            "worktree_entry_count": self.worktree_entry_count,
            "worktree_status_digest": self.worktree_status_digest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible representation."""

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "name": self.name,
            "purpose": self.purpose,
            "checkout": self.checkout_reference,
            "baseline_commit": self.baseline_commit,
            "head_commit": self.head_commit,
            "baseline_matches": self.baseline_matches,
            "branch": self.branch,
            "remotes": dict(sorted(self.remotes.items())),
            "missing_files": list(self.missing_files),
            "missing_directories": list(self.missing_directories),
            "worktree_state": self.worktree_state,
            "worktree_entry_count": self.worktree_entry_count,
            "worktree_status_digest": self.worktree_status_digest,
            "reproducibility_state": self.reproducibility_state,
            "source_digest": self.source_digest,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _validate_anchor_paths(
    source_name: str,
    paths: tuple[PurePosixPath, ...],
    *,
    kind: str,
) -> None:
    rendered: list[str] = []
    for path in paths:
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise SourceConfigurationError(
                f"source {source_name!r} has an unsafe required {kind}: {path}"
            )
        rendered.append(path.as_posix())
    if len(rendered) != len(set(rendered)):
        raise SourceConfigurationError(
            f"source {source_name!r} has duplicate required {kind} anchors"
        )


def normalize_repository_url(value: str) -> str:
    """Normalize common GitHub SSH/HTTPS remote forms for identity comparison."""

    candidate = value.strip()
    if not candidate:
        return ""
    if candidate.startswith("git@") and ":" in candidate:
        host_and_user, path = candidate.split(":", 1)
        host = host_and_user.split("@", 1)[1]
        candidate = f"{host}/{path}"
    elif "://" in candidate:
        parsed = urlsplit(candidate)
        host = parsed.hostname or ""
        candidate = f"{host}/{parsed.path.lstrip('/')}"
    candidate = candidate.removesuffix(".git").rstrip("/")
    return candidate.casefold()


def _as_string_tuple(raw: object, *, field_name: str, source_name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise SourceConfigurationError(
            f"source {source_name!r} field {field_name!r} must be an array of strings"
        )
    return tuple(cast(list[str], raw))


def load_source_specs(
    lock_path: Path = DEFAULT_LOCK_PATH,
    *,
    checkout_overrides: Mapping[str, Path] | None = None,
) -> tuple[SourceSpec, ...]:
    """Load and validate the versioned source-baseline lock."""

    try:
        decoded = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise SourceConfigurationError(f"cannot read source lock {lock_path}: {error}") from error
    if decoded.get("schema_version") != SCHEMA_VERSION:
        raise SourceConfigurationError(
            f"unsupported source lock schema: {decoded.get('schema_version')!r}"
        )
    raw_sources = decoded.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise SourceConfigurationError("source lock must declare at least one source")

    overrides = checkout_overrides or {}
    specs: list[SourceSpec] = []
    for raw_value in raw_sources:
        if not isinstance(raw_value, dict):
            raise SourceConfigurationError("each source lock entry must be a table")
        raw = cast(dict[str, object], raw_value)
        name = str(raw.get("name", ""))
        checkout_reference = str(raw.get("checkout", ""))
        override = overrides.get(name)
        checkout = (
            override.expanduser().resolve()
            if override is not None
            else (REPOSITORY_ROOT / checkout_reference).resolve()
        )
        if override is not None:
            checkout_reference = f"<override:{name}>"
        raw_remotes = raw.get("remotes")
        if not isinstance(raw_remotes, dict):
            raise SourceConfigurationError(f"source {name!r} remotes must be a table")
        remotes = {str(key): str(value) for key, value in raw_remotes.items()}
        required_files = tuple(
            PurePosixPath(path)
            for path in _as_string_tuple(
                raw.get("required_files", []),
                field_name="required_files",
                source_name=name,
            )
        )
        required_directories = tuple(
            PurePosixPath(path)
            for path in _as_string_tuple(
                raw.get("required_directories", []),
                field_name="required_directories",
                source_name=name,
            )
        )
        specs.append(
            SourceSpec(
                name=name,
                purpose=str(raw.get("purpose", "")),
                checkout=checkout,
                checkout_reference=checkout_reference,
                baseline_commit=str(raw.get("baseline_commit", "")),
                remotes=remotes,
                required_files=required_files,
                required_directories=required_directories,
            )
        )
    names = tuple(spec.name for spec in specs)
    if len(names) != len(set(names)):
        raise SourceConfigurationError("source lock names must be unique")
    unknown_overrides = set(overrides) - set(names)
    if unknown_overrides:
        raise SourceConfigurationError(
            f"checkout overrides refer to unknown sources: {sorted(unknown_overrides)}"
        )
    return tuple(specs)


def _git_text(
    client: GitClient,
    repository: Path,
    arguments: tuple[str, ...],
    *,
    timeout_seconds: float,
) -> GitCommandResult:
    return client.run(repository, arguments, timeout_seconds=timeout_seconds)


def inspect_source_checkout(
    spec: SourceSpec,
    *,
    client: GitClient | None = None,
    include_worktree: bool = False,
    require_clean: bool = False,
    timeout_seconds: float = 120.0,
) -> SourceCheckoutReport:
    """Inspect one checkout without importing or executing its software."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    actual_client = client or SubprocessGitClient()
    errors: list[str] = []
    warnings: list[str] = []
    head_commit: str | None = None
    branch: str | None = None
    actual_remotes: dict[str, str] = {}
    worktree_state = "not_checked"
    worktree_entry_count: int | None = None
    worktree_status_digest: str | None = None

    if not spec.checkout.is_dir():
        errors.append("checkout directory is missing")
    else:
        inside = _git_text(
            actual_client,
            spec.checkout,
            ("rev-parse", "--is-inside-work-tree"),
            timeout_seconds=timeout_seconds,
        )
        if inside.return_code != 0 or inside.stdout.strip() != "true":
            errors.append("checkout is not a Git worktree")
        else:
            head = _git_text(
                actual_client,
                spec.checkout,
                ("rev-parse", "HEAD"),
                timeout_seconds=timeout_seconds,
            )
            if head.return_code != 0:
                errors.append("cannot resolve checkout HEAD")
            else:
                head_commit = head.stdout.strip()
                if head_commit != spec.baseline_commit:
                    errors.append(
                        f"HEAD {head_commit or '<empty>'} does not match locked baseline "
                        f"{spec.baseline_commit}"
                    )
            branch_result = _git_text(
                actual_client,
                spec.checkout,
                ("symbolic-ref", "--quiet", "--short", "HEAD"),
                timeout_seconds=timeout_seconds,
            )
            if branch_result.return_code == 0:
                branch = branch_result.stdout.strip() or None
            else:
                warnings.append("checkout is detached or its branch cannot be resolved")

            for remote_name, expected_repository in spec.remotes.items():
                remote = _git_text(
                    actual_client,
                    spec.checkout,
                    ("remote", "get-url", remote_name),
                    timeout_seconds=timeout_seconds,
                )
                if remote.return_code != 0:
                    errors.append(f"required Git remote {remote_name!r} is missing")
                    continue
                normalized = normalize_repository_url(remote.stdout)
                actual_remotes[remote_name] = normalized
                if normalized != expected_repository:
                    errors.append(
                        f"remote {remote_name!r} points to {normalized!r}, expected "
                        f"{expected_repository!r}"
                    )

            if include_worktree or require_clean:
                status = _git_text(
                    actual_client,
                    spec.checkout,
                    ("status", "--porcelain=v1", "--untracked-files=all"),
                    timeout_seconds=timeout_seconds,
                )
                if status.timed_out:
                    worktree_state = "timed_out"
                    warnings.append("worktree scan timed out")
                    if require_clean:
                        errors.append("clean worktree was required but could not be established")
                elif status.return_code != 0:
                    worktree_state = "error"
                    errors.append("cannot inspect worktree state")
                else:
                    status_text = status.stdout
                    worktree_entry_count = len(status_text.splitlines())
                    worktree_status_digest = hashlib.sha256(status_text.encode("utf-8")).hexdigest()
                    worktree_state = "dirty" if status_text else "clean"
                    if require_clean and worktree_state != "clean":
                        errors.append(
                            f"clean worktree was required but {worktree_entry_count} change(s) exist"
                        )

    missing_files = tuple(
        path.as_posix()
        for path in spec.required_files
        if not spec.checkout.joinpath(*path.parts).is_file()
    )
    missing_directories = tuple(
        path.as_posix()
        for path in spec.required_directories
        if not spec.checkout.joinpath(*path.parts).is_dir()
    )
    if missing_files:
        errors.append(f"missing {len(missing_files)} required source file(s)")
    if missing_directories:
        errors.append(f"missing {len(missing_directories)} required source directory/directories")

    return SourceCheckoutReport(
        name=spec.name,
        purpose=spec.purpose,
        checkout_reference=spec.checkout_reference,
        baseline_commit=spec.baseline_commit,
        head_commit=head_commit,
        branch=branch,
        remotes=MappingProxyType(actual_remotes),
        missing_files=missing_files,
        missing_directories=missing_directories,
        worktree_state=worktree_state,
        worktree_entry_count=worktree_entry_count,
        worktree_status_digest=worktree_status_digest,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the development source-checkout CLI parser."""

    parser = argparse.ArgumentParser(
        description="validate locked Elmer and FDTDX development source checkouts"
    )
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--elmer", type=Path, help="override the locked Elmer checkout")
    parser.add_argument("--fdtdx", type=Path, help="override the locked FDTDX checkout")
    parser.add_argument(
        "--source",
        action="append",
        choices=("elmer", "fdtdx"),
        default=[],
        help="inspect only one named source; may be repeated",
    )
    parser.add_argument(
        "--include-worktree",
        action="store_true",
        help="also inspect tracked and untracked worktree changes; may be slow",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="fail unless every worktree is clean; implies --include-worktree",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=120.0,
        help="per-Git-command timeout",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the source-checkout gate."""

    args = build_parser().parse_args(argv)
    overrides = {
        name: path
        for name, path in (("elmer", args.elmer), ("fdtdx", args.fdtdx))
        if path is not None
    }
    try:
        specs = load_source_specs(args.lock.resolve(), checkout_overrides=overrides)
        selected_names = frozenset(args.source)
        selected_specs = tuple(
            spec for spec in specs if not selected_names or spec.name in selected_names
        )
        reports = tuple(
            inspect_source_checkout(
                spec,
                include_worktree=args.include_worktree,
                require_clean=args.require_clean,
                timeout_seconds=args.timeout_seconds,
            )
            for spec in selected_specs
        )
    except (SourceConfigurationError, ValueError) as config_error:
        print(f"source checkout configuration error: {config_error}", file=sys.stderr)
        return 2

    valid = all(report.valid for report in reports)
    if args.json:
        payload = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "valid": valid,
            "sources": [report.to_dict() for report in reports],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for report in reports:
            status = "PASS" if report.valid else "FAIL"
            head = report.head_commit[:12] if report.head_commit is not None else "unavailable"
            branch = report.branch or "detached/unknown"
            print(
                f"{report.name:8} {status:4} head={head} branch={branch} "
                f"worktree={report.worktree_state} digest={report.source_digest[:12]}"
            )
            for issue in report.errors:
                print(f"  error: {issue}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
        print("source checkout gate passed" if valid else "source checkout gate failed")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
