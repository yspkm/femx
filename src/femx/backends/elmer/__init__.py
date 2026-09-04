"""Dependency-light Elmer process-boundary utilities.

Numerical adapters remain explicit submodule imports so the base package does not import NumPy.
No Elmer source or physics module is included in femx.
"""

from femx.backends.elmer.runner import (
    ElmerCommand,
    ElmerInstallation,
    ElmerProcessResult,
    ElmerRunner,
)

__all__ = [
    "ElmerCommand",
    "ElmerInstallation",
    "ElmerProcessResult",
    "ElmerRunner",
]
