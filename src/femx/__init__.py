"""Public API for the femx architecture harness."""

from femx.backends.protocol import Backend, PreparedProblem, PrepareRequest, SolveRequest
from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.problem import MeshSpec, ObservableSpec, PhysicsSpec, Problem
from femx.core.solution import ConvergenceReport, Field, Solution
from femx.runtime import prepare, solve

__all__ = [
    "AnalysisKind",
    "Backend",
    "CapabilityRequest",
    "CapabilitySet",
    "ConvergenceReport",
    "Field",
    "FunctionSpaceFamily",
    "GradientMethod",
    "MeshSpec",
    "ObservableSpec",
    "ParallelModel",
    "PhysicsSpec",
    "PrepareRequest",
    "PreparedProblem",
    "Problem",
    "ScalarKind",
    "Solution",
    "SolveRequest",
    "prepare",
    "solve",
]

__version__ = "0.1.0.dev1"
