"""Compatibility names for the port collective runtime evidence schema.

The sharding mechanics are element-family independent and live in
``femx.backends.jax.collective_runtime``.  Existing port evidence keeps its original schema and
public names so historical process records remain verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding

from .collective_runtime import (
    CollectiveAddressableShard,
    CollectiveArrayReport,
    CollectiveCompilerMemoryReport,
    CollectiveTimingReport,
    _leading_partition,
    collective_named_sharding,
    describe_collective_array,
    make_collective_array_from_process_local_data,
)

PORT_COLLECTIVE_ARRAY_REPORT_SCHEMA = "femx.jax.port_collective.array_report/v1"
PORT_COLLECTIVE_MEMORY_REPORT_SCHEMA = "femx.jax.port_collective.memory_report/v1"
PORT_COLLECTIVE_TIMING_REPORT_SCHEMA = "femx.jax.port_collective.timing_report/v1"

PortCollectiveAddressableShard = CollectiveAddressableShard


@dataclass(frozen=True, slots=True)
class PortCollectiveArrayReport(CollectiveArrayReport):
    """Historical port-specific view of a generic collective array report."""

    schema_version: str = PORT_COLLECTIVE_ARRAY_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = PORT_COLLECTIVE_ARRAY_REPORT_SCHEMA


@dataclass(frozen=True, slots=True)
class PortCollectiveCompilerMemoryReport(CollectiveCompilerMemoryReport):
    """Historical port-specific view of compiler memory evidence."""

    schema_version: str = PORT_COLLECTIVE_MEMORY_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = PORT_COLLECTIVE_MEMORY_REPORT_SCHEMA


@dataclass(frozen=True, slots=True)
class PortCollectiveTimingReport(CollectiveTimingReport):
    """Historical port-specific view of phase-separated timing evidence."""

    schema_version: str = PORT_COLLECTIVE_TIMING_REPORT_SCHEMA
    EXPECTED_SCHEMA_VERSION: ClassVar[str] = PORT_COLLECTIVE_TIMING_REPORT_SCHEMA


def _port_report(report: CollectiveArrayReport) -> PortCollectiveArrayReport:
    return PortCollectiveArrayReport(
        name=report.name,
        global_shape=report.global_shape,
        dtype=report.dtype,
        partition_axis_name=report.partition_axis_name,
        partition_count=report.partition_count,
        global_device_count=report.global_device_count,
        process_index=report.process_index,
        process_count=report.process_count,
        global_logical_bytes=report.global_logical_bytes,
        addressable_logical_bytes=report.addressable_logical_bytes,
        addressable_shards=report.addressable_shards,
    )


def collective_port_named_sharding(
    mesh: Mesh,
    rank: int,
    *,
    axis_name: str = "partition",
) -> NamedSharding:
    """Return the unchanged port compatibility name for leading-axis sharding."""

    return collective_named_sharding(mesh, rank, axis_name=axis_name)


def describe_collective_port_array(
    name: str,
    array: jax.Array,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> PortCollectiveArrayReport:
    """Describe a port array while preserving its historical schema id."""

    return _port_report(describe_collective_array(name, array, mesh, axis_name=axis_name))


def make_collective_port_array_from_process_local_data(
    name: str,
    global_host_array: np.ndarray,
    mesh: Mesh,
    *,
    axis_name: str = "partition",
) -> tuple[jax.Array, PortCollectiveArrayReport]:
    """Create a port array through the solver-neutral process-local loader."""

    array, report = make_collective_array_from_process_local_data(
        name,
        global_host_array,
        mesh,
        axis_name=axis_name,
    )
    return array, _port_report(report)


__all__ = [
    "PORT_COLLECTIVE_ARRAY_REPORT_SCHEMA",
    "PORT_COLLECTIVE_MEMORY_REPORT_SCHEMA",
    "PORT_COLLECTIVE_TIMING_REPORT_SCHEMA",
    "PortCollectiveAddressableShard",
    "PortCollectiveArrayReport",
    "PortCollectiveCompilerMemoryReport",
    "PortCollectiveTimingReport",
    "_leading_partition",
    "collective_port_named_sharding",
    "describe_collective_port_array",
    "make_collective_port_array_from_process_local_data",
]
