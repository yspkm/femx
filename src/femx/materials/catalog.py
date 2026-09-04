"""Provenance-first, solver-neutral material catalog contracts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from itertools import pairwise
from pathlib import Path
from typing import TypeAlias, cast

from femx.core.errors import ContractError

ModelValue: TypeAlias = float | tuple[float, ...]

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


class MaterialError(ContractError):
    """A material record, selection, or evaluation is not scientifically valid."""


class PropertyStatus(StrEnum):
    """Permitted use of one curated property record."""

    EXECUTABLE = "executable"
    REFERENCE_ONLY = "reference_only"
    REQUIRES_CALIBRATION = "requires_calibration"


class ModelKind(StrEnum):
    """Small set of numerical models implemented by the v1 catalog."""

    CONSTANT = "constant"
    SELLMEIER = "sellmeier"
    LINEAR_TABLE = "linear_table"
    REFERENCE = "reference"


def _trimmed(value: str, *, label: str) -> str:
    if not value or value.strip() != value:
        raise MaterialError(f"{label} must be non-empty and trimmed")
    return value


def _identifier(value: str, *, label: str) -> str:
    _trimmed(value, label=label)
    if not _IDENTIFIER.fullmatch(value):
        raise MaterialError(f"{label} must be a lowercase stable identifier")
    return value


def _finite(value: float, *, label: str) -> float:
    if not math.isfinite(value):
        raise MaterialError(f"{label} must be finite")
    return value


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MaterialError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise MaterialError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _list(value: object, *, label: str) -> list[object]:
    if not isinstance(value, list):
        raise MaterialError(f"{label} must be an array")
    return cast(list[object], value)


def _string(value: object, *, label: str) -> str:
    if not isinstance(value, str):
        raise MaterialError(f"{label} must be a string")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterialError(f"{label} must be a JSON number")
    return float(value)


def _strings(value: object, *, label: str) -> tuple[str, ...]:
    values = _list(value, label=label)
    return tuple(_trimmed(_string(item, label=label), label=label) for item in values)


@dataclass(frozen=True, slots=True)
class ClosedInterval:
    """Closed SI interval used to reject missing inputs and extrapolation."""

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        _finite(self.minimum, label="interval minimum")
        _finite(self.maximum, label="interval maximum")
        if self.minimum > self.maximum:
            raise MaterialError("interval minimum cannot exceed maximum")

    def to_dict(self) -> dict[str, float]:
        """Return the JSON representation."""

        return {"minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ClosedInterval:
        """Decode one interval from JSON data."""

        return cls(
            minimum=_number(data["minimum"], label="interval minimum"),
            maximum=_number(data["maximum"], label="interval maximum"),
        )


@dataclass(frozen=True, slots=True)
class PropertyValidity:
    """Independent-variable and specimen validity carried by a property."""

    temperature_k: ClosedInterval | None = None
    vacuum_wavelength_m: ClosedInterval | None = None
    carrier_concentration_m3: ClosedInterval | None = None
    photon_energy_ev: ClosedInterval | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for note in self.notes:
            _trimmed(note, label="validity note")

    def require_inputs(
        self,
        *,
        temperature_k: float | None,
        vacuum_wavelength_m: float | None,
        carrier_concentration_m3: float | None,
        photon_energy_ev: float | None,
    ) -> None:
        """Reject absent inputs and values outside every declared interval."""

        for label, value, interval in (
            ("temperature_k", temperature_k, self.temperature_k),
            ("vacuum_wavelength_m", vacuum_wavelength_m, self.vacuum_wavelength_m),
            (
                "carrier_concentration_m3",
                carrier_concentration_m3,
                self.carrier_concentration_m3,
            ),
            ("photon_energy_ev", photon_energy_ev, self.photon_energy_ev),
        ):
            if interval is None:
                continue
            if value is None:
                raise MaterialError(f"{label} is required by the property validity contract")
            _finite(value, label=label)
            if not interval.minimum <= value <= interval.maximum:
                raise MaterialError(
                    f"{label}={value!r} is outside [{interval.minimum}, {interval.maximum}]"
                )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "temperature_k": None if self.temperature_k is None else self.temperature_k.to_dict(),
            "vacuum_wavelength_m": (
                None if self.vacuum_wavelength_m is None else self.vacuum_wavelength_m.to_dict()
            ),
            "carrier_concentration_m3": (
                None
                if self.carrier_concentration_m3 is None
                else self.carrier_concentration_m3.to_dict()
            ),
            "photon_energy_ev": (
                None if self.photon_energy_ev is None else self.photon_energy_ev.to_dict()
            ),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PropertyValidity:
        """Decode a validity contract."""

        def interval(name: str) -> ClosedInterval | None:
            raw = data.get(name)
            if raw is None:
                return None
            return ClosedInterval.from_dict(_mapping(raw, label=f"validity {name}"))

        return cls(
            temperature_k=interval("temperature_k"),
            vacuum_wavelength_m=interval("vacuum_wavelength_m"),
            carrier_concentration_m3=interval("carrier_concentration_m3"),
            photon_energy_ev=interval("photon_energy_ev"),
            notes=_strings(data.get("notes", []), label="validity notes"),
        )


@dataclass(frozen=True, slots=True)
class MaterialCitation:
    """Exact bibliographic identity used by one or more property records."""

    citation_id: str
    authors: tuple[str, ...]
    title: str
    container: str
    year: int
    doi: str | None
    url: str
    locator: str

    def __post_init__(self) -> None:
        _identifier(self.citation_id, label="citation_id")
        if not self.authors:
            raise MaterialError("citation must include at least one author or institution")
        for author in self.authors:
            _trimmed(author, label="citation author")
        _trimmed(self.title, label="citation title")
        _trimmed(self.container, label="citation container")
        if not 1800 <= self.year <= 2100:
            raise MaterialError("citation year is outside the supported range")
        if self.doi is not None and not self.doi.startswith("10."):
            raise MaterialError("citation DOI must start with '10.'")
        if not self.url.startswith("https://"):
            raise MaterialError("citation URL must use HTTPS")
        _trimmed(self.locator, label="citation locator")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "id": self.citation_id,
            "authors": list(self.authors),
            "title": self.title,
            "container": self.container,
            "year": self.year,
            "doi": self.doi,
            "url": self.url,
            "locator": self.locator,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MaterialCitation:
        """Decode a citation."""

        raw_doi = data.get("doi")
        raw_year = data["year"]
        if isinstance(raw_year, bool) or not isinstance(raw_year, int):
            raise MaterialError("citation year must be an integer")
        return cls(
            citation_id=_string(data["id"], label="citation id"),
            authors=_strings(data["authors"], label="citation authors"),
            title=_string(data["title"], label="citation title"),
            container=_string(data["container"], label="citation container"),
            year=raw_year,
            doi=None if raw_doi is None else _string(raw_doi, label="citation DOI"),
            url=_string(data["url"], label="citation URL"),
            locator=_string(data["locator"], label="citation locator"),
        )


@dataclass(frozen=True, slots=True)
class ModelParameter:
    """One named scalar or one-dimensional coefficient vector."""

    name: str
    value: ModelValue

    def __post_init__(self) -> None:
        _identifier(self.name, label="model parameter name")
        if isinstance(self.value, tuple):
            if not self.value:
                raise MaterialError(f"model parameter {self.name!r} cannot be empty")
            for item in self.value:
                _finite(item, label=f"model parameter {self.name!r}")
        else:
            _finite(self.value, label=f"model parameter {self.name!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        value: object = list(self.value) if isinstance(self.value, tuple) else self.value
        return {"name": self.name, "value": value}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ModelParameter:
        """Decode a model parameter."""

        raw_value = data["value"]
        value: ModelValue
        if isinstance(raw_value, list):
            value = tuple(_number(item, label="model parameter value") for item in raw_value)
        else:
            value = _number(raw_value, label="model parameter value")
        return cls(name=_string(data["name"], label="model parameter name"), value=value)


@dataclass(frozen=True, slots=True)
class PropertyModel:
    """One explicitly implemented or deliberately non-executable property model."""

    kind: ModelKind
    parameters: tuple[ModelParameter, ...] = ()

    def __post_init__(self) -> None:
        names = tuple(parameter.name for parameter in self.parameters)
        if len(names) != len(set(names)):
            raise MaterialError("model parameter names must be unique")
        expected = {
            ModelKind.CONSTANT: {"value"},
            ModelKind.SELLMEIER: {"b", "resonance_wavelength_um"},
            ModelKind.LINEAR_TABLE: {"temperature_k", "values"},
            ModelKind.REFERENCE: set(),
        }[self.kind]
        if set(names) != expected:
            raise MaterialError(
                f"{self.kind.value} model requires parameters {sorted(expected)}, got {sorted(names)}"
            )
        if self.kind is ModelKind.CONSTANT and isinstance(self.parameter("value"), tuple):
            raise MaterialError("constant model value must be scalar")
        if self.kind is ModelKind.SELLMEIER:
            b = self.vector("b")
            resonance = self.vector("resonance_wavelength_um")
            if len(b) != len(resonance):
                raise MaterialError("Sellmeier coefficient vectors must have equal length")
            if any(value <= 0 for value in resonance):
                raise MaterialError("Sellmeier resonance wavelengths must be positive")
        if self.kind is ModelKind.LINEAR_TABLE:
            axis = self.vector("temperature_k")
            values = self.vector("values")
            if len(axis) != len(values) or len(axis) < 2:
                raise MaterialError("linear table axes must have equal length of at least two")
            if any(right <= left for left, right in pairwise(axis)):
                raise MaterialError("linear table temperature axis must be strictly increasing")

    def parameter(self, name: str) -> ModelValue:
        """Return one parameter by exact name."""

        for parameter in self.parameters:
            if parameter.name == name:
                return parameter.value
        raise MaterialError(f"model parameter {name!r} is absent")

    def vector(self, name: str) -> tuple[float, ...]:
        """Return one vector parameter and reject a scalar substitution."""

        value = self.parameter(name)
        if not isinstance(value, tuple):
            raise MaterialError(f"model parameter {name!r} must be a vector")
        return value

    def evaluate(
        self,
        *,
        temperature_k: float | None,
        vacuum_wavelength_m: float | None,
    ) -> float:
        """Evaluate a supported scalar model without extrapolation policy."""

        if self.kind is ModelKind.REFERENCE:
            raise MaterialError("reference model has no executable numerical representation")
        if self.kind is ModelKind.CONSTANT:
            value = self.parameter("value")
            if isinstance(value, tuple):  # pragma: no cover - protected by construction
                raise MaterialError("constant model value must be scalar")
            return value
        if self.kind is ModelKind.SELLMEIER:
            if vacuum_wavelength_m is None:
                raise MaterialError("vacuum_wavelength_m is required by the Sellmeier model")
            wavelength_um = vacuum_wavelength_m * 1.0e6
            wavelength_squared = wavelength_um * wavelength_um
            refractive_index_squared = 1.0
            for coefficient, resonance_um in zip(
                self.vector("b"), self.vector("resonance_wavelength_um"), strict=True
            ):
                denominator = wavelength_squared - resonance_um * resonance_um
                if denominator == 0.0:
                    raise MaterialError("Sellmeier model is singular at a resonance wavelength")
                refractive_index_squared += coefficient * wavelength_squared / denominator
            if refractive_index_squared <= 0.0 or not math.isfinite(refractive_index_squared):
                raise MaterialError("Sellmeier model produced a nonphysical refractive index")
            return math.sqrt(refractive_index_squared)
        if temperature_k is None:
            raise MaterialError("temperature_k is required by the linear table")
        axis = self.vector("temperature_k")
        values = self.vector("values")
        if temperature_k == axis[-1]:
            return values[-1]
        upper = bisect_right(axis, temperature_k)
        if upper == 0 or upper == len(axis):
            raise MaterialError("linear table evaluation would extrapolate")
        lower = upper - 1
        fraction = (temperature_k - axis[lower]) / (axis[upper] - axis[lower])
        return values[lower] + fraction * (values[upper] - values[lower])

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "kind": self.kind.value,
            "parameters": [parameter.to_dict() for parameter in self.parameters],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> PropertyModel:
        """Decode a property model."""

        raw_parameters = _list(data.get("parameters", []), label="model parameters")
        return cls(
            kind=ModelKind(_string(data["kind"], label="model kind")),
            parameters=tuple(
                ModelParameter.from_dict(_mapping(item, label="model parameter"))
                for item in raw_parameters
            ),
        )


@dataclass(frozen=True, slots=True)
class MaterialProperty:
    """One property with model, usage status, validity, and citations."""

    property_id: str
    quantity: str
    unit: str
    status: PropertyStatus
    model: PropertyModel
    validity: PropertyValidity
    citation_ids: tuple[str, ...]
    uncertainty: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.property_id, label="property_id")
        _identifier(self.quantity, label="property quantity")
        _trimmed(self.unit, label="property unit")
        if not self.citation_ids:
            raise MaterialError("material property must cite at least one source")
        for citation_id in self.citation_ids:
            _identifier(citation_id, label="property citation id")
        if len(self.citation_ids) != len(set(self.citation_ids)):
            raise MaterialError("property citation ids must be unique")
        if self.uncertainty is not None:
            _trimmed(self.uncertainty, label="property uncertainty")
        for note in self.notes:
            _trimmed(note, label="property note")
        if self.status is PropertyStatus.EXECUTABLE and self.model.kind is ModelKind.REFERENCE:
            raise MaterialError("an executable property must have a numerical model")
        if (
            self.status is not PropertyStatus.EXECUTABLE
            and self.model.kind is not ModelKind.REFERENCE
        ):
            raise MaterialError("non-executable properties must not expose a numerical model")

    def evaluate(
        self,
        *,
        temperature_k: float | None = None,
        vacuum_wavelength_m: float | None = None,
        carrier_concentration_m3: float | None = None,
        photon_energy_ev: float | None = None,
    ) -> float:
        """Evaluate only an executable model inside every declared validity interval."""

        if self.status is not PropertyStatus.EXECUTABLE:
            raise MaterialError(
                f"property {self.property_id!r} is {self.status.value} and cannot be evaluated"
            )
        self.validity.require_inputs(
            temperature_k=temperature_k,
            vacuum_wavelength_m=vacuum_wavelength_m,
            carrier_concentration_m3=carrier_concentration_m3,
            photon_energy_ev=photon_energy_ev,
        )
        return self.model.evaluate(
            temperature_k=temperature_k,
            vacuum_wavelength_m=vacuum_wavelength_m,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "id": self.property_id,
            "quantity": self.quantity,
            "unit": self.unit,
            "status": self.status.value,
            "model": self.model.to_dict(),
            "validity": self.validity.to_dict(),
            "citation_ids": list(self.citation_ids),
            "uncertainty": self.uncertainty,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MaterialProperty:
        """Decode a property record."""

        raw_uncertainty = data.get("uncertainty")
        return cls(
            property_id=_string(data["id"], label="property id"),
            quantity=_string(data["quantity"], label="property quantity"),
            unit=_string(data["unit"], label="property unit"),
            status=PropertyStatus(_string(data["status"], label="property status")),
            model=PropertyModel.from_dict(_mapping(data["model"], label="property model")),
            validity=PropertyValidity.from_dict(
                _mapping(data["validity"], label="property validity")
            ),
            citation_ids=_strings(data["citation_ids"], label="property citation ids"),
            uncertainty=(
                None
                if raw_uncertainty is None
                else _string(raw_uncertainty, label="property uncertainty")
            ),
            notes=_strings(data.get("notes", []), label="property notes"),
        )


@dataclass(frozen=True, slots=True)
class MaterialRecord:
    """A specimen/process-aware material or explicitly doped material family member."""

    material_id: str
    name: str
    formula: str
    phase: str
    form: str
    process: str
    aliases: tuple[str, ...]
    properties: tuple[MaterialProperty, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.material_id, label="material_id")
        for label, value in (
            ("material name", self.name),
            ("material formula", self.formula),
            ("material phase", self.phase),
            ("material form", self.form),
            ("material process", self.process),
        ):
            _trimmed(value, label=label)
        for alias in self.aliases:
            _trimmed(alias, label="material alias")
        if len({alias.casefold() for alias in self.aliases}) != len(self.aliases):
            raise MaterialError("material aliases must be unique ignoring case")
        if not self.properties:
            raise MaterialError("material record must contain at least one property")
        property_ids = tuple(prop.property_id for prop in self.properties)
        if len(property_ids) != len(set(property_ids)):
            raise MaterialError("material property ids must be unique")
        for note in self.notes:
            _trimmed(note, label="material note")

    def property(self, property_id: str) -> MaterialProperty:
        """Return one exact property record."""

        for prop in self.properties:
            if prop.property_id == property_id:
                return prop
        raise MaterialError(f"material {self.material_id!r} has no property {property_id!r}")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON representation."""

        return {
            "id": self.material_id,
            "name": self.name,
            "formula": self.formula,
            "phase": self.phase,
            "form": self.form,
            "process": self.process,
            "aliases": list(self.aliases),
            "properties": [prop.to_dict() for prop in self.properties],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MaterialRecord:
        """Decode a material record."""

        raw_properties = _list(data["properties"], label="material properties")
        return cls(
            material_id=_string(data["id"], label="material id"),
            name=_string(data["name"], label="material name"),
            formula=_string(data["formula"], label="material formula"),
            phase=_string(data["phase"], label="material phase"),
            form=_string(data["form"], label="material form"),
            process=_string(data["process"], label="material process"),
            aliases=_strings(data.get("aliases", []), label="material aliases"),
            properties=tuple(
                MaterialProperty.from_dict(_mapping(item, label="material property"))
                for item in raw_properties
            ),
            notes=_strings(data.get("notes", []), label="material notes"),
        )


@dataclass(frozen=True, slots=True)
class MaterialCatalog:
    """Versioned catalog whose canonical representation is content-addressable."""

    catalog_id: str
    catalog_version: str
    citations: tuple[MaterialCitation, ...]
    materials: tuple[MaterialRecord, ...]
    schema_version: str = "femx.material_catalog/v1"

    def __post_init__(self) -> None:
        _identifier(self.catalog_id, label="catalog_id")
        _trimmed(self.catalog_version, label="catalog_version")
        if self.schema_version != "femx.material_catalog/v1":
            raise MaterialError(f"unsupported material catalog schema {self.schema_version!r}")
        citation_ids = tuple(citation.citation_id for citation in self.citations)
        if len(citation_ids) != len(set(citation_ids)):
            raise MaterialError("catalog citation ids must be unique")
        material_ids = tuple(material.material_id for material in self.materials)
        if len(material_ids) != len(set(material_ids)):
            raise MaterialError("catalog material ids must be unique")
        citation_set = set(citation_ids)
        selectors: dict[str, str] = {}
        for material in self.materials:
            for selector in (material.material_id, *material.aliases):
                normalized = selector.casefold()
                previous = selectors.get(normalized)
                if previous is not None and previous != material.material_id:
                    raise MaterialError(
                        f"material selector {selector!r} is shared by {previous!r} "
                        f"and {material.material_id!r}"
                    )
                selectors[normalized] = material.material_id
            for prop in material.properties:
                missing = set(prop.citation_ids) - citation_set
                if missing:
                    raise MaterialError(
                        f"property {material.material_id}.{prop.property_id} cites unknown "
                        f"sources {sorted(missing)}"
                    )

    def citation(self, citation_id: str) -> MaterialCitation:
        """Return one citation by stable id."""

        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation
        raise MaterialError(f"catalog has no citation {citation_id!r}")

    def material(self, selector: str) -> MaterialRecord:
        """Resolve a material id or explicit alias without fuzzy matching."""

        normalized = selector.strip().casefold()
        for material in self.materials:
            if normalized in {
                material.material_id.casefold(),
                *(alias.casefold() for alias in material.aliases),
            }:
                return material
        raise MaterialError(f"catalog has no material selector {selector!r}")

    def property(self, material: str, property_id: str) -> MaterialProperty:
        """Resolve one property from one material selection."""

        return self.material(material).property(property_id)

    def evaluate(
        self,
        material: str,
        property_id: str,
        *,
        temperature_k: float | None = None,
        vacuum_wavelength_m: float | None = None,
        carrier_concentration_m3: float | None = None,
        photon_energy_ev: float | None = None,
    ) -> float:
        """Evaluate a selected property under explicit independent variables."""

        return self.property(material, property_id).evaluate(
            temperature_k=temperature_k,
            vacuum_wavelength_m=vacuum_wavelength_m,
            carrier_concentration_m3=carrier_concentration_m3,
            photon_energy_ev=photon_energy_ev,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the complete versioned JSON representation."""

        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "catalog_version": self.catalog_version,
            "citations": [citation.to_dict() for citation in self.citations],
            "materials": [material.to_dict() for material in self.materials],
        }

    def canonical_json(self) -> str:
        """Serialize deterministically for provenance and cache keys."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def digest(self) -> str:
        """Return SHA-256 of the complete canonical catalog."""

        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> MaterialCatalog:
        """Decode and validate one material catalog."""

        raw_citations = _list(data["citations"], label="catalog citations")
        raw_materials = _list(data["materials"], label="catalog materials")
        return cls(
            schema_version=_string(data["schema_version"], label="catalog schema version"),
            catalog_id=_string(data["catalog_id"], label="catalog id"),
            catalog_version=_string(data["catalog_version"], label="catalog version"),
            citations=tuple(
                MaterialCitation.from_dict(_mapping(item, label="catalog citation"))
                for item in raw_citations
            ),
            materials=tuple(
                MaterialRecord.from_dict(_mapping(item, label="catalog material"))
                for item in raw_materials
            ),
        )

    @classmethod
    def from_json(cls, encoded: str) -> MaterialCatalog:
        """Decode JSON while rejecting duplicate keys and non-finite constants."""

        def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, value in pairs:
                if key in result:
                    raise MaterialError(f"duplicate JSON key {key!r}")
                result[key] = value
            return result

        def reject_constant(value: str) -> object:
            raise MaterialError(f"non-finite JSON constant {value!r} is forbidden")

        try:
            decoded = json.loads(
                encoded,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except json.JSONDecodeError as error:
            raise MaterialError(f"invalid material catalog JSON: {error.msg}") from error
        try:
            return cls.from_dict(_mapping(decoded, label="material catalog"))
        except MaterialError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise MaterialError(f"invalid material catalog structure: {error}") from error


@lru_cache(maxsize=1)
def builtin_catalog() -> MaterialCatalog:
    """Load the immutable femx v1 catalog packaged with the distribution."""

    encoded = (
        resources.files("femx.materials").joinpath("catalog-v1.json").read_text(encoding="utf-8")
    )
    return MaterialCatalog.from_json(encoded)


def load_material_catalog(path: Path) -> MaterialCatalog:
    """Load a caller-selected catalog file without implicit path discovery."""

    try:
        encoded = path.read_text(encoding="utf-8")
    except OSError as error:
        raise MaterialError(f"cannot read material catalog {path}: {error}") from error
    return MaterialCatalog.from_json(encoded)
