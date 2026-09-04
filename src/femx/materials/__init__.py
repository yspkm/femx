"""Provenance-first material records and external compatibility imports."""

from femx.materials.catalog import (
    ClosedInterval,
    MaterialCatalog,
    MaterialCitation,
    MaterialError,
    MaterialProperty,
    MaterialRecord,
    ModelKind,
    ModelParameter,
    PropertyModel,
    PropertyStatus,
    PropertyValidity,
    builtin_catalog,
    load_material_catalog,
)
from femx.materials.elmer import (
    ElmerLegacyMaterial,
    ElmerLegacyMaterialLibrary,
    ElmerLegacyParameter,
    load_elmer_material_library,
)

__all__ = [
    "ClosedInterval",
    "ElmerLegacyMaterial",
    "ElmerLegacyMaterialLibrary",
    "ElmerLegacyParameter",
    "MaterialCatalog",
    "MaterialCitation",
    "MaterialError",
    "MaterialProperty",
    "MaterialRecord",
    "ModelKind",
    "ModelParameter",
    "PropertyModel",
    "PropertyStatus",
    "PropertyValidity",
    "builtin_catalog",
    "load_elmer_material_library",
    "load_material_catalog",
]
