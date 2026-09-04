"""Deterministic test doubles shared by contract tests."""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

from femx.backends.protocol import (
    BackendDescriptor,
    PreparedProblem,
    PrepareRequest,
    SolveRequest,
)
from femx.core.capabilities import (
    AnalysisKind,
    CapabilityRequest,
    CapabilitySet,
    FunctionSpaceFamily,
    GradientMethod,
    ParallelModel,
    ScalarKind,
)
from femx.core.problem import Problem
from femx.core.solution import ConvergenceReport, ConvergenceStatus, Solution
from femx.mesh import CellType, EntityTag, Mesh, MeshGeometry, MeshTopology


@dataclass(frozen=True, slots=True)
class FakeDType:
    """Minimal dtype descriptor."""

    kind: str = "f"


class FakeArray:
    """Minimal array satisfying ``ArrayLike`` without NumPy."""

    def __init__(self, shape: tuple[int, ...], *, dtype_kind: str = "f") -> None:
        self._shape = shape
        self._dtype = FakeDType(dtype_kind)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def dtype(self) -> FakeDType:
        return self._dtype


def dummy_mesh() -> Mesh:
    """Return a minimal valid triangular mesh contract."""

    return Mesh(
        geometry=MeshGeometry(FakeArray((3, 2))),
        topology=MeshTopology(FakeArray((1, 3)), CellType.TRIANGLE, node_count=3),
    )


def structured_unit_square_mesh(intervals: int) -> Mesh:
    """Return a deterministic P1 triangle mesh with explicit, tagged boundary segments."""

    if intervals <= 0:
        raise ValueError("intervals must be positive")
    width = intervals + 1
    coordinates = np.asarray(
        [(i / intervals, j / intervals) for j in range(width) for i in range(width)],
        dtype=np.float64,
    )

    def node(i: int, j: int) -> int:
        return j * width + i

    cells: list[tuple[int, int, int]] = []
    for j in range(intervals):
        for i in range(intervals):
            lower_left = node(i, j)
            lower_right = node(i + 1, j)
            upper_left = node(i, j + 1)
            upper_right = node(i + 1, j + 1)
            cells.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )

    facets: list[tuple[int, int]] = []
    tag_ids: dict[str, tuple[int, ...]] = {}
    for name, edges in (
        ("bottom", [(node(i, 0), node(i + 1, 0)) for i in range(intervals)]),
        (
            "right",
            [(node(intervals, j), node(intervals, j + 1)) for j in range(intervals)],
        ),
        (
            "top",
            [(node(i + 1, intervals), node(i, intervals)) for i in range(intervals)],
        ),
        ("left", [(node(0, j + 1), node(0, j)) for j in range(intervals)]),
    ):
        start = len(facets)
        facets.extend(edges)
        tag_ids[name] = tuple(range(start, len(facets)))

    cell_array = np.asarray(cells, dtype=np.int32)
    facet_array = np.asarray(facets, dtype=np.int32)
    tags = (
        EntityTag("domain", 2, tuple(range(cell_array.shape[0]))),
        *(EntityTag(name, 1, ids) for name, ids in tag_ids.items()),
    )
    return Mesh(
        geometry=MeshGeometry(coordinates),
        topology=MeshTopology(cell_array, CellType.TRIANGLE, coordinates.shape[0]),
        tags=tags,
        boundary_facets=MeshTopology(facet_array, CellType.SEGMENT, coordinates.shape[0]),
    )


@dataclass(frozen=True, slots=True)
class DummyPhysics:
    """Small valid physics specification for backend harness tests."""

    kind: str = "steady_heat"
    requirements: CapabilityRequest = field(
        default_factory=lambda: CapabilityRequest(
            analysis=AnalysisKind.STEADY,
            function_spaces=frozenset({FunctionSpaceFamily.H1}),
        )
    )
    valid: bool = True

    def validate(self) -> None:
        if not self.valid:
            raise ValueError("invalid dummy physics")

    def canonical_data(self) -> Mapping[str, object]:
        return {"kind": self.kind}


class FakeBackend:
    """A deterministic in-memory backend used only by the contract harness."""

    descriptor = BackendDescriptor(name="fake", version="1.0")
    capabilities = CapabilitySet(
        analyses=frozenset({AnalysisKind.STEADY}),
        function_spaces=frozenset({FunctionSpaceFamily.H1}),
        scalar_kinds=frozenset({ScalarKind.REAL}),
        gradients=frozenset({GradientMethod.NONE}),
        parallel_models=frozenset({ParallelModel.SERIAL}),
    )

    def __init__(self) -> None:
        self.prepare_calls = 0
        self.solve_calls = 0

    def prepare(self, problem: Problem, request: PrepareRequest) -> PreparedProblem:
        self.prepare_calls += 1
        return PreparedProblem(backend=self.descriptor, problem=problem, payload={"prepared": True})

    def solve(self, prepared: PreparedProblem, request: SolveRequest) -> Solution:
        self.solve_calls += 1
        return Solution(
            backend_name=self.descriptor.name,
            backend_version=self.descriptor.version,
            fields={},
            observables={},
            convergence=ConvergenceReport(status=ConvergenceStatus.CONVERGED, iterations=1),
        )
