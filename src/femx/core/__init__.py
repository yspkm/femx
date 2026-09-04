"""Backend-neutral semantic contracts."""

from femx.core.axes import Axis, AxisDirection, Direction
from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.execution import ExecutionPolicy
from femx.core.parameters import (
    ParameterReference,
    ParameterRole,
    ParameterSchema,
    ParameterSpec,
    ParameterValues,
)
from femx.core.problem import MeshSpec, ObservableSpec, PhysicsSpec, Problem
from femx.core.solution import ConvergenceReport, Field, Solution

__all__ = [
    "AnalysisKind",
    "Axis",
    "AxisDirection",
    "CapabilityRequest",
    "CapabilitySet",
    "ConvergenceReport",
    "Direction",
    "ExecutionPolicy",
    "Field",
    "FunctionSpaceFamily",
    "GradientMethod",
    "MeshSpec",
    "ObservableSpec",
    "ParallelModel",
    "ParameterReference",
    "ParameterRole",
    "ParameterSchema",
    "ParameterSpec",
    "ParameterValues",
    "PhysicsSpec",
    "Problem",
    "ScalarKind",
    "Solution",
]
