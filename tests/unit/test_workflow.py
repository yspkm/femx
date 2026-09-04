import pytest

from femx.core.errors import ContractError
from femx.workflows import CouplingEdge, WorkflowGraph, WorkflowNode

pytestmark = pytest.mark.unit


def edge(source: str, quantity: str, target: str, target_quantity: str) -> CouplingEdge:
    return CouplingEdge(
        source,
        quantity,
        target,
        target_quantity,
        operator="conservative_transfer/v1",
        source_unit="K",
        target_unit="K",
    )


def test_workflow_graph_produces_deterministic_topological_order() -> None:
    graph = WorkflowGraph(
        nodes=(
            WorkflowNode("optical", frozenset({"temperature"}), frozenset({"mode"})),
            WorkflowNode("thermal", frozenset({"power"}), frozenset({"temperature"})),
            WorkflowNode("current", frozenset(), frozenset({"power"})),
        ),
        edges=(
            edge("current", "power", "thermal", "power"),
            edge("thermal", "temperature", "optical", "temperature"),
        ),
    )

    assert graph.topological_order() == ("current", "thermal", "optical")


def test_workflow_graph_rejects_cycles_and_missing_quantities() -> None:
    a = WorkflowNode("a", frozenset({"y"}), frozenset({"x"}))
    b = WorkflowNode("b", frozenset({"x"}), frozenset({"y"}))
    with pytest.raises(ContractError, match="cycle"):
        WorkflowGraph((a, b), (edge("a", "x", "b", "x"), edge("b", "y", "a", "y")))
    with pytest.raises(ContractError, match="does not produce"):
        WorkflowGraph((a, b), (edge("a", "missing", "b", "x"),))


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: WorkflowNode(" bad", frozenset(), frozenset()), "node name"),
        (lambda: WorkflowNode("bad", frozenset({""}), frozenset()), "empty quantity"),
        (
            lambda: CouplingEdge("a", "x", "a", "y", "op", "K", "K"),
            "own source",
        ),
        (
            lambda: CouplingEdge("", "x", "b", "y", "op", "K", "K"),
            "non-empty",
        ),
        (lambda: WorkflowGraph((), ()), "at least one node"),
    ],
)
def test_workflow_schema_rejects_ambiguous_nodes_and_edges(factory, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        factory()


def test_workflow_graph_rejects_duplicate_unknown_and_multi_producer_edges() -> None:
    source_a = WorkflowNode("a", frozenset(), frozenset({"x"}))
    source_b = WorkflowNode("b", frozenset(), frozenset({"x"}))
    target = WorkflowNode("target", frozenset({"input"}), frozenset())
    with pytest.raises(ContractError, match="node names"):
        WorkflowGraph((source_a, source_a), ())
    with pytest.raises(ContractError, match="unknown node"):
        WorkflowGraph((source_a,), (edge("a", "x", "missing", "input"),))
    with pytest.raises(ContractError, match="does not consume"):
        WorkflowGraph((source_a, target), (edge("a", "x", "target", "missing"),))
    with pytest.raises(ContractError, match="multiple producers"):
        WorkflowGraph(
            (source_a, source_b, target),
            (
                edge("a", "x", "target", "input"),
                edge("b", "x", "target", "input"),
            ),
        )
