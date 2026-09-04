"""Read-only compatibility import for an external ElmerGUI material XML file."""

from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from femx.materials.catalog import MaterialError

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class ElmerLegacyParameter:
    """Whitespace-normalized XML character data plus an optional exact scalar parse."""

    name: str
    raw_value: str
    scalar_value: float | None

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise MaterialError("Elmer parameter name must be non-empty and trimmed")
        if not self.raw_value or self.raw_value.strip() != self.raw_value:
            raise MaterialError("Elmer parameter value must be non-empty and trimmed")
        if self.scalar_value is not None and not math.isfinite(self.scalar_value):
            raise MaterialError("Elmer scalar parameter must be finite")

    def to_dict(self) -> dict[str, object]:
        """Return the compatibility representation."""

        return {
            "name": self.name,
            "raw_value": self.raw_value,
            "scalar_value": self.scalar_value,
        }


@dataclass(frozen=True, slots=True)
class ElmerLegacyMaterial:
    """One material copied in memory from the explicitly selected XML source."""

    name: str
    parameters: tuple[ElmerLegacyParameter, ...]

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise MaterialError("Elmer material name must be non-empty and trimmed")
        names = tuple(parameter.name for parameter in self.parameters)
        if not names:
            raise MaterialError("Elmer material must contain at least one parameter")
        if len(names) != len(set(names)):
            raise MaterialError(f"Elmer material {self.name!r} has duplicate parameters")

    def parameter(self, name: str) -> ElmerLegacyParameter:
        """Return one parameter by exact Elmer name."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter
        raise MaterialError(f"Elmer material {self.name!r} has no parameter {name!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the compatibility representation."""

        return {
            "name": self.name,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }


@dataclass(frozen=True, slots=True)
class ElmerLegacyMaterialLibrary:
    """Content-addressed snapshot view of one separately installed ElmerGUI XML file."""

    source_path: str
    source_sha256: str
    source_revision: str
    materials: tuple[ElmerLegacyMaterial, ...]
    usage_status: str = "legacy_unverified"
    schema_version: str = "femx.elmer_gui_materials/v1"

    def __post_init__(self) -> None:
        path = Path(self.source_path)
        if not path.is_absolute():
            raise MaterialError("Elmer material source path must be absolute")
        if not _SHA256.fullmatch(self.source_sha256):
            raise MaterialError("Elmer material source hash must be a lowercase SHA-256")
        if not self.source_revision or self.source_revision.strip() != self.source_revision:
            raise MaterialError("Elmer source revision must be non-empty and trimmed")
        if self.usage_status != "legacy_unverified":
            raise MaterialError("Elmer GUI values must remain legacy_unverified")
        if self.schema_version != "femx.elmer_gui_materials/v1":
            raise MaterialError(f"unsupported Elmer material schema {self.schema_version!r}")
        names = tuple(material.name for material in self.materials)
        if not names:
            raise MaterialError("Elmer material selection cannot be empty")
        if len(names) != len(set(names)):
            raise MaterialError("Elmer material names must be unique")

    def material(self, name: str) -> ElmerLegacyMaterial:
        """Return one exact Elmer material without fuzzy aliases."""

        for material in self.materials:
            if material.name == name:
                return material
        raise MaterialError(f"Elmer library has no selected material {name!r}")

    def to_dict(self) -> dict[str, object]:
        """Return complete provenance and selected source values."""

        return {
            "schema_version": self.schema_version,
            "usage_status": self.usage_status,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "source_revision": self.source_revision,
            "materials": [material.to_dict() for material in self.materials],
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for run provenance."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """Hash selected values and source identity without the machine-local path."""

        identity = self.to_dict()
        del identity["source_path"]
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _scalar_value(raw_value: str) -> float | None:
    try:
        value = float(raw_value)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def load_elmer_material_library(
    path: Path,
    *,
    source_revision: str,
    selected_names: tuple[str, ...] | None = None,
) -> ElmerLegacyMaterialLibrary:
    """Load selected ElmerGUI materials without vendoring or interpreting MATC."""

    resolved = path.resolve()
    try:
        encoded = resolved.read_bytes()
    except OSError as error:
        raise MaterialError(f"cannot read Elmer material library {resolved}: {error}") from error
    try:
        root = ET.fromstring(encoded)
    except ET.ParseError as error:
        raise MaterialError(f"invalid Elmer material XML: {error}") from error
    if root.tag != "materiallibrary":
        raise MaterialError(f"expected materiallibrary root, got {root.tag!r}")

    parsed: list[ElmerLegacyMaterial] = []
    seen_materials: set[str] = set()
    for element in root:
        if element.tag != "material":
            raise MaterialError(f"unexpected Elmer material-library element {element.tag!r}")
        name = (element.get("name") or "").strip()
        if not name:
            raise MaterialError("Elmer material is missing a name")
        if name in seen_materials:
            raise MaterialError(f"duplicate Elmer material {name!r}")
        seen_materials.add(name)
        parameters: list[ElmerLegacyParameter] = []
        seen_parameters: set[str] = set()
        for child in element:
            if child.tag != "parameter":
                raise MaterialError(f"unexpected element in Elmer material {name!r}: {child.tag!r}")
            parameter_name = (child.get("name") or "").strip()
            raw_value = (child.text or "").strip()
            if not parameter_name or not raw_value:
                raise MaterialError(f"Elmer material {name!r} has an incomplete parameter")
            if parameter_name in seen_parameters:
                raise MaterialError(
                    f"Elmer material {name!r} has duplicate parameter {parameter_name!r}"
                )
            seen_parameters.add(parameter_name)
            parameters.append(
                ElmerLegacyParameter(
                    name=parameter_name,
                    raw_value=raw_value,
                    scalar_value=_scalar_value(raw_value),
                )
            )
        parsed.append(ElmerLegacyMaterial(name=name, parameters=tuple(parameters)))

    if selected_names is not None:
        if not selected_names or len(selected_names) != len(set(selected_names)):
            raise MaterialError("selected Elmer material names must be non-empty and unique")
        requested = set(selected_names)
        missing = requested - seen_materials
        if missing:
            raise MaterialError(f"Elmer material selection is missing {sorted(missing)}")
        parsed = [material for material in parsed if material.name in requested]

    return ElmerLegacyMaterialLibrary(
        source_path=str(resolved),
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        source_revision=source_revision,
        materials=tuple(parsed),
    )
