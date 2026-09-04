"""Acyclic multiphysics workflow contracts."""

from dataclasses import dataclass

from femx.core.errors import ContractError


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    """One solve or deterministic transformation in a workflow."""

    name: str
    consumes: frozenset[str]
    produces: frozenset[str]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ContractError("workflow node name must be non-empty and trimmed")
        if any(not item for item in (*self.consumes, *self.produces)):
            raise ContractError(f"workflow node {self.name!r} has an empty quantity name")


@dataclass(frozen=True, slots=True)
class CouplingEdge:
    """An explicit quantity transfer and its numerical operator identity."""

    source: str
    source_quantity: str
    target: str
    target_quantity: str
    operator: str
    source_unit: str
    target_unit: str

    def __post_init__(self) -> None:
        values = (
            self.source,
            self.source_quantity,
            self.target,
            self.target_quantity,
            self.operator,
            self.source_unit,
            self.target_unit,
        )
        if any(not value for value in values):
            raise ContractError("coupling edge fields must be non-empty")
        if self.source == self.target:
            raise ContractError("a coupling edge cannot target its own source node")


@dataclass(frozen=True, slots=True)
class WorkflowGraph:
    """A validated directed acyclic graph of multiphysics operations."""

    nodes: tuple[WorkflowNode, ...]
    edges: tuple[CouplingEdge, ...]

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ContractError("workflow graph must contain at least one node")
        by_name = {node.name: node for node in self.nodes}
        if len(by_name) != len(self.nodes):
            raise ContractError("workflow node names must be unique")

        claimed_inputs: set[tuple[str, str]] = set()
        for edge in self.edges:
            if edge.source not in by_name or edge.target not in by_name:
                raise ContractError(
                    f"coupling edge references an unknown node: {edge.source!r} -> {edge.target!r}"
                )
            if edge.source_quantity not in by_name[edge.source].produces:
                raise ContractError(
                    f"node {edge.source!r} does not produce {edge.source_quantity!r}"
                )
            if edge.target_quantity not in by_name[edge.target].consumes:
                raise ContractError(
                    f"node {edge.target!r} does not consume {edge.target_quantity!r}"
                )
            target_input = (edge.target, edge.target_quantity)
            if target_input in claimed_inputs:
                raise ContractError(
                    f"workflow input {edge.target!r}.{edge.target_quantity} has multiple producers"
                )
            claimed_inputs.add(target_input)
        self.topological_order()

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic order or raise if the graph contains a cycle."""

        names = {node.name for node in self.nodes}
        incoming = {name: 0 for name in names}
        outgoing: dict[str, set[str]] = {name: set() for name in names}
        for edge in self.edges:
            if edge.target not in outgoing[edge.source]:
                outgoing[edge.source].add(edge.target)
                incoming[edge.target] += 1

        ready = sorted(name for name, count in incoming.items() if count == 0)
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for target in sorted(outgoing[current]):
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()

        if len(ordered) != len(names):
            cyclic = sorted(name for name, count in incoming.items() if count > 0)
            raise ContractError(f"workflow graph contains a cycle involving {cyclic}")
        return tuple(ordered)
