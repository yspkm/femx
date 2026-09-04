"""Guarded Gmsh generation and strict MSH ingestion."""

from femx.meshing.gmsh.importer import (
    GmshImportRecord,
    GmshPhysicalGroup,
    ImportedGmshMesh,
    read_gmsh_msh,
    read_gmsh_msh_3d,
)
from femx.meshing.gmsh.recipes import RectangularWaveguideCrossSection
from femx.meshing.gmsh.ring_heater import (
    PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY,
    PUBLIC_TIDY3D_NOTEBOOK_REVISION,
    PUBLIC_TIDY3D_NOTEBOOK_SHA256,
    PUBLIC_TIDY3D_RING_PAGE,
    RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA,
    PublicRingHeater3D,
    RingHeaterMeshProfile,
    RingHeaterThermalSensitivity3D,
    ring_heater_mesh_profile,
)
from femx.meshing.gmsh.ring_heater_quality import (
    PublicRingHeaterMeshReport,
    evaluate_public_ring_heater_mesh,
)
from femx.meshing.gmsh.runner import (
    GmshInstallation,
    GmshMeshingRequest,
    GmshProcessResult,
    GmshRunner,
    GmshToolIdentity,
)

__all__ = [
    "PUBLIC_TIDY3D_NOTEBOOK_REPOSITORY",
    "PUBLIC_TIDY3D_NOTEBOOK_REVISION",
    "PUBLIC_TIDY3D_NOTEBOOK_SHA256",
    "PUBLIC_TIDY3D_RING_PAGE",
    "RING_HEATER_THERMAL_SENSITIVITY_GEOMETRY_SCHEMA",
    "GmshImportRecord",
    "GmshInstallation",
    "GmshMeshingRequest",
    "GmshPhysicalGroup",
    "GmshProcessResult",
    "GmshRunner",
    "GmshToolIdentity",
    "ImportedGmshMesh",
    "PublicRingHeater3D",
    "PublicRingHeaterMeshReport",
    "RectangularWaveguideCrossSection",
    "RingHeaterMeshProfile",
    "RingHeaterThermalSensitivity3D",
    "evaluate_public_ring_heater_mesh",
    "read_gmsh_msh",
    "read_gmsh_msh_3d",
    "ring_heater_mesh_profile",
]
