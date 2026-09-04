"""Typed multiphysics coupling graphs."""

from femx.workflows.electrothermal import (
    JOULE_HEAT_DENSITY_UNIT,
    SAME_MESH_CELL_LOCAL_P1_OPERATOR,
    SAME_MESH_CELL_OPERATOR,
    CoupledIterationPolicy,
    ResistivityTemperatureLaw,
    SameMeshJouleHeating,
    SelfConsistentJouleHeating,
)
from femx.workflows.graph import CouplingEdge, WorkflowGraph, WorkflowNode

__all__ = [
    "JOULE_HEAT_DENSITY_UNIT",
    "SAME_MESH_CELL_LOCAL_P1_OPERATOR",
    "SAME_MESH_CELL_OPERATOR",
    "CoupledIterationPolicy",
    "CouplingEdge",
    "ResistivityTemperatureLaw",
    "SameMeshJouleHeating",
    "SelfConsistentJouleHeating",
    "WorkflowGraph",
    "WorkflowNode",
]
