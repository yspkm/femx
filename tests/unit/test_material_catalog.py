import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from femx.materials import (
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

pytestmark = pytest.mark.unit


def citation(citation_id: str = "source") -> MaterialCitation:
    return MaterialCitation(
        citation_id=citation_id,
        authors=("A. Author",),
        title="Measured property",
        container="Journal 1, 1-2",
        year=2020,
        doi="10.1234/example",
        url="https://doi.org/10.1234/example",
        locator="Table 1",
    )


def constant_model(value: float = 2.0) -> PropertyModel:
    return PropertyModel(
        ModelKind.CONSTANT,
        (ModelParameter("value", value),),
    )


def material_property(
    property_id: str = "thermal.value",
    *,
    status: PropertyStatus = PropertyStatus.EXECUTABLE,
    model: PropertyModel | None = None,
    citation_ids: tuple[str, ...] = ("source",),
) -> MaterialProperty:
    return MaterialProperty(
        property_id=property_id,
        quantity="thermal_conductivity",
        unit="W m^-1 K^-1",
        status=status,
        model=constant_model() if model is None else model,
        validity=PropertyValidity(),
        citation_ids=citation_ids,
        uncertainty="one sigma",
        notes=("test specimen",),
    )


def material(
    material_id: str = "test_material",
    *,
    aliases: tuple[str, ...] = ("Test material",),
    properties: tuple[MaterialProperty, ...] | None = None,
) -> MaterialRecord:
    return MaterialRecord(
        material_id=material_id,
        name="Test material",
        formula="X",
        phase="solid",
        form="test coupon",
        process="documented process",
        aliases=aliases,
        properties=(material_property(),) if properties is None else properties,
        notes=("record note",),
    )


def catalog(*, materials: tuple[MaterialRecord, ...] | None = None) -> MaterialCatalog:
    return MaterialCatalog(
        catalog_id="test.catalog",
        catalog_version="1",
        citations=(citation(),),
        materials=(material(),) if materials is None else materials,
    )


def test_builtin_catalog_is_content_addressed_and_resolves_explicit_aliases() -> None:
    first = builtin_catalog()
    second = builtin_catalog()

    assert first is second
    assert len(first.materials) == 11
    assert len(first.citations) == 18
    assert first.material("c-Si").material_id == "si_crystalline_intrinsic"
    assert first.material("  FUSED SILICA  ").material_id == "sio2_fused"
    assert first.citation("malitson1965").doi == "10.1364/JOSA.55.001205"
    assert len(first.digest()) == 64
    assert MaterialCatalog.from_json(first.canonical_json()) == first


def test_builtin_material_ids_and_authoritative_source_identities_are_explicit() -> None:
    builtins = builtin_catalog()

    assert {record.material_id for record in builtins.materials} == {
        "al_reference",
        "cu_reference",
        "ge_crystalline",
        "si_crystalline_intrinsic",
        "si_n_arsenic",
        "si_n_phosphorus",
        "si_p_boron",
        "sio2_fused",
        "sio2_thermal",
        "ti_reference",
        "tin_reference",
    }
    assert {source.citation_id: source.doi for source in builtins.citations} == {
        "cocorullo1999": "10.1063/1.123337",
        "glassbrenner_slack1964": "10.1103/PhysRev.134.A1058",
        "green2008": "10.1016/j.solmat.2008.06.009",
        "herzinger1998": "10.1063/1.367101",
        "hust_lankford1984": "10.6028/NBS.IR.84-3007",
        "kearney2018": "10.1016/j.tsf.2018.07.001",
        "li1980": "10.1063/1.555624",
        "malitson1965": "10.1364/JOSA.55.001205",
        "masetti1983": "10.1109/T-ED.1983.21207",
        "nist_janaf_si1998": None,
        "nist_alloy_data2019": "10.18434/M32153",
        "nist_scd_z00752": "10.18434/T4F30D",
        "nunley2016": "10.1116/1.4963075",
        "powell_tye1961": "10.1016/0022-5088(61)90064-9",
        "rakic1998": "10.1364/AO.37.005271",
        "reddy2017": "10.1021/acsphotonics.7b00127",
        "soref_bennett1987": "10.1109/JQE.1987.1073206",
        "taylor_morreale1964": "10.1111/j.1151-2916.1964.tb15657.x",
    }


def test_malitson_model_matches_the_published_1550_nm_value_and_guards_range() -> None:
    builtins = builtin_catalog()
    model = builtins.property("sio2_fused", "optical.refractive_index.malitson1965").model
    assert model.vector("b") == (0.6961663, 0.4079426, 0.8974794)
    assert model.vector("resonance_wavelength_um") == (0.0684043, 0.1162414, 9.896161)
    value = builtins.evaluate(
        "sio2_fused",
        "optical.refractive_index.malitson1965",
        temperature_k=293.15,
        vacuum_wavelength_m=1.55e-6,
    )

    assert value == pytest.approx(1.4440236217032607, rel=1.0e-14)
    with pytest.raises(MaterialError, match="temperature_k is required"):
        builtins.evaluate(
            "sio2_fused",
            "optical.refractive_index.malitson1965",
            vacuum_wavelength_m=1.55e-6,
        )
    with pytest.raises(MaterialError, match="outside"):
        builtins.evaluate(
            "sio2_fused",
            "optical.refractive_index.malitson1965",
            temperature_k=293.15,
            vacuum_wavelength_m=4.0e-6,
        )


def test_tin_nist_table_and_specimen_density_are_exact_and_never_extrapolate() -> None:
    builtins = builtin_catalog()
    model = builtins.property("tin_reference", "thermal.heat_capacity.nist_z00752").model

    assert model.vector("temperature_k") == tuple(float(value) for value in range(300, 1801, 100))
    assert model.vector("values") == (
        601.71,
        705.31,
        756.67,
        787.48,
        808.57,
        824.49,
        837.41,
        848.45,
        858.28,
        867.27,
        875.68,
        883.66,
        891.33,
        898.76,
        906.0,
        913.11,
    )

    assert builtins.evaluate("tin_reference", "mass_density.nist_z00752") == 5240.0
    assert (
        builtins.evaluate("tin_reference", "thermal.heat_capacity.nist_z00752", temperature_k=300.0)
        == 601.71
    )
    assert builtins.evaluate(
        "tin_reference", "thermal.heat_capacity.nist_z00752", temperature_k=350.0
    ) == pytest.approx((601.71 + 705.31) / 2.0)
    assert (
        builtins.evaluate(
            "tin_reference", "thermal.heat_capacity.nist_z00752", temperature_k=1800.0
        )
        == 913.11
    )
    with pytest.raises(MaterialError, match="outside"):
        builtins.evaluate("tin_reference", "thermal.heat_capacity.nist_z00752", temperature_k=299.0)


def test_reference_and_calibration_records_cannot_be_evaluated() -> None:
    builtins = builtin_catalog()

    assert (
        builtins.property("ge_crystalline", "optical.refractive_index.li1980").status
        is PropertyStatus.REFERENCE_ONLY
    )
    with pytest.raises(MaterialError, match="reference_only"):
        builtins.evaluate(
            "ge_crystalline",
            "optical.refractive_index.li1980",
            temperature_k=300.0,
            vacuum_wavelength_m=1.55e-6,
        )
    with pytest.raises(MaterialError, match="requires_calibration"):
        builtins.evaluate(
            "tin_reference",
            "optical.complex_permittivity.reddy2017",
            vacuum_wavelength_m=1.55e-6,
        )


def test_catalog_file_loader_round_trips_and_reports_missing_files(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    path.write_text(catalog().canonical_json(), encoding="utf-8")

    assert load_material_catalog(path) == catalog()
    with pytest.raises(MaterialError, match="cannot read material catalog"):
        load_material_catalog(tmp_path / "missing.json")


def test_closed_interval_and_validity_reject_nonfinite_reversed_missing_and_outside_values() -> (
    None
):
    interval = ClosedInterval(1.0, 2.0)
    assert ClosedInterval.from_dict(interval.to_dict()) == interval
    with pytest.raises(MaterialError, match="finite"):
        ClosedInterval(math.nan, 2.0)
    with pytest.raises(MaterialError, match="finite"):
        ClosedInterval(1.0, math.inf)
    with pytest.raises(MaterialError, match="cannot exceed"):
        ClosedInterval(2.0, 1.0)

    validity = PropertyValidity(
        temperature_k=interval,
        vacuum_wavelength_m=interval,
        carrier_concentration_m3=interval,
        photon_energy_ev=interval,
        notes=("all inputs explicit",),
    )
    validity.require_inputs(
        temperature_k=1.5,
        vacuum_wavelength_m=1.5,
        carrier_concentration_m3=1.5,
        photon_energy_ev=1.5,
    )
    assert PropertyValidity.from_dict(validity.to_dict()) == validity

    for field in (
        "temperature_k",
        "vacuum_wavelength_m",
        "carrier_concentration_m3",
        "photon_energy_ev",
    ):
        values = {
            "temperature_k": 1.5,
            "vacuum_wavelength_m": 1.5,
            "carrier_concentration_m3": 1.5,
            "photon_energy_ev": 1.5,
        }
        values[field] = None
        with pytest.raises(MaterialError, match=f"{field} is required"):
            validity.require_inputs(**values)  # type: ignore[arg-type]
        values[field] = 3.0
        with pytest.raises(MaterialError, match=f"{field}=3.0.*outside"):
            validity.require_inputs(**values)  # type: ignore[arg-type]
        values[field] = math.nan
        with pytest.raises(MaterialError, match="finite"):
            validity.require_inputs(**values)  # type: ignore[arg-type]

    with pytest.raises(MaterialError, match="validity note"):
        PropertyValidity(notes=(" bad ",))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"citation_id": "UPPER"}, "stable identifier"),
        ({"authors": ()}, "at least one author"),
        ({"authors": (" bad ",)}, "citation author"),
        ({"title": ""}, "citation title"),
        ({"container": ""}, "citation container"),
        ({"year": 1799}, "year"),
        ({"year": 2101}, "year"),
        ({"doi": "bad"}, "DOI"),
        ({"url": "http://example.test"}, "HTTPS"),
        ({"locator": ""}, "locator"),
    ],
)
def test_citation_validation(override: dict[str, object], message: str) -> None:
    with pytest.raises(MaterialError, match=message):
        replace(citation(), **override)


def test_citation_and_model_parameter_serialization() -> None:
    source = citation()
    assert MaterialCitation.from_dict(source.to_dict()) == source

    scalar = ModelParameter("value", 2.0)
    vector = ModelParameter("values", (1.0, 2.0))
    assert ModelParameter.from_dict(scalar.to_dict()) == scalar
    assert ModelParameter.from_dict(vector.to_dict()) == vector
    with pytest.raises(MaterialError, match="stable identifier"):
        ModelParameter("Bad", 1.0)
    with pytest.raises(MaterialError, match="cannot be empty"):
        ModelParameter("values", ())
    with pytest.raises(MaterialError, match="finite"):
        ModelParameter("value", math.nan)
    with pytest.raises(MaterialError, match="finite"):
        ModelParameter("values", (1.0, math.inf))
    with pytest.raises(MaterialError, match="JSON number"):
        ModelParameter.from_dict({"name": "value", "value": True})
    with pytest.raises(MaterialError, match="JSON number"):
        ModelParameter.from_dict({"name": "value", "value": "2.0"})
    with pytest.raises(MaterialError, match="JSON number"):
        ModelParameter.from_dict({"name": "values", "value": [1.0, False]})


def test_property_model_schema_validation() -> None:
    with pytest.raises(MaterialError, match="unique"):
        PropertyModel(
            ModelKind.SELLMEIER,
            (ModelParameter("b", (1.0,)), ModelParameter("b", (2.0,))),
        )
    with pytest.raises(MaterialError, match="requires parameters"):
        PropertyModel(ModelKind.REFERENCE, (ModelParameter("value", 1.0),))
    with pytest.raises(MaterialError, match="must be scalar"):
        PropertyModel(ModelKind.CONSTANT, (ModelParameter("value", (1.0,)),))
    with pytest.raises(MaterialError, match="equal length"):
        PropertyModel(
            ModelKind.SELLMEIER,
            (
                ModelParameter("b", (1.0, 2.0)),
                ModelParameter("resonance_wavelength_um", (1.0,)),
            ),
        )
    with pytest.raises(MaterialError, match="must be positive"):
        PropertyModel(
            ModelKind.SELLMEIER,
            (
                ModelParameter("b", (1.0,)),
                ModelParameter("resonance_wavelength_um", (0.0,)),
            ),
        )
    with pytest.raises(MaterialError, match="equal length of at least two"):
        PropertyModel(
            ModelKind.LINEAR_TABLE,
            (
                ModelParameter("temperature_k", (1.0,)),
                ModelParameter("values", (2.0,)),
            ),
        )
    with pytest.raises(MaterialError, match="strictly increasing"):
        PropertyModel(
            ModelKind.LINEAR_TABLE,
            (
                ModelParameter("temperature_k", (1.0, 1.0)),
                ModelParameter("values", (2.0, 3.0)),
            ),
        )


def test_property_model_evaluation_failure_modes() -> None:
    reference = PropertyModel(ModelKind.REFERENCE)
    with pytest.raises(MaterialError, match="no executable"):
        reference.evaluate(temperature_k=None, vacuum_wavelength_m=None)
    with pytest.raises(MaterialError, match="absent"):
        reference.parameter("missing")
    with pytest.raises(MaterialError, match="must be a vector"):
        constant_model().vector("value")

    sellmeier = PropertyModel(
        ModelKind.SELLMEIER,
        (
            ModelParameter("b", (1.0,)),
            ModelParameter("resonance_wavelength_um", (1.0,)),
        ),
    )
    with pytest.raises(MaterialError, match="wavelength_m is required"):
        sellmeier.evaluate(temperature_k=None, vacuum_wavelength_m=None)
    with pytest.raises(MaterialError, match="singular"):
        sellmeier.evaluate(temperature_k=None, vacuum_wavelength_m=1.0e-6)

    nonphysical = PropertyModel(
        ModelKind.SELLMEIER,
        (
            ModelParameter("b", (-2.0,)),
            ModelParameter("resonance_wavelength_um", (1.0,)),
        ),
    )
    with pytest.raises(MaterialError, match="nonphysical"):
        nonphysical.evaluate(temperature_k=None, vacuum_wavelength_m=2.0e-6)

    table = PropertyModel(
        ModelKind.LINEAR_TABLE,
        (
            ModelParameter("temperature_k", (1.0, 2.0)),
            ModelParameter("values", (10.0, 20.0)),
        ),
    )
    with pytest.raises(MaterialError, match="temperature_k is required"):
        table.evaluate(temperature_k=None, vacuum_wavelength_m=None)
    with pytest.raises(MaterialError, match="extrapolate"):
        table.evaluate(temperature_k=0.0, vacuum_wavelength_m=None)
    with pytest.raises(MaterialError, match="extrapolate"):
        table.evaluate(temperature_k=3.0, vacuum_wavelength_m=None)
    assert table.evaluate(temperature_k=1.5, vacuum_wavelength_m=None) == 15.0
    assert table.evaluate(temperature_k=2.0, vacuum_wavelength_m=None) == 20.0
    assert PropertyModel.from_dict(table.to_dict()) == table


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"property_id": "Bad"}, "stable identifier"),
        ({"quantity": "Bad"}, "stable identifier"),
        ({"unit": ""}, "unit"),
        ({"citation_ids": ()}, "at least one source"),
        ({"citation_ids": ("source", "source")}, "unique"),
        ({"citation_ids": ("Bad",)}, "stable identifier"),
        ({"uncertainty": " bad "}, "uncertainty"),
        ({"notes": (" bad ",)}, "property note"),
        (
            {"status": PropertyStatus.EXECUTABLE, "model": PropertyModel(ModelKind.REFERENCE)},
            "must have a numerical model",
        ),
        (
            {"status": PropertyStatus.REFERENCE_ONLY, "model": constant_model()},
            "must not expose",
        ),
    ],
)
def test_material_property_validation(override: dict[str, object], message: str) -> None:
    with pytest.raises(MaterialError, match=message):
        replace(material_property(), **override)


def test_material_property_serialization_and_validity_forwarding() -> None:
    prop = material_property()
    assert MaterialProperty.from_dict(prop.to_dict()) == prop
    assert prop.evaluate() == 2.0

    reference = material_property(
        status=PropertyStatus.REFERENCE_ONLY,
        model=PropertyModel(ModelKind.REFERENCE),
    )
    with pytest.raises(MaterialError, match="reference_only"):
        reference.evaluate()


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"material_id": "Bad"}, "stable identifier"),
        ({"name": ""}, "material name"),
        ({"formula": ""}, "formula"),
        ({"phase": ""}, "phase"),
        ({"form": ""}, "form"),
        ({"process": ""}, "process"),
        ({"aliases": ("same", "Same")}, "aliases"),
        ({"properties": ()}, "at least one property"),
        (
            {"properties": (material_property(), material_property())},
            "property ids must be unique",
        ),
        ({"notes": (" bad ",)}, "material note"),
    ],
)
def test_material_record_validation(override: dict[str, object], message: str) -> None:
    with pytest.raises(MaterialError, match=message):
        replace(material(), **override)


def test_material_record_serialization_and_missing_property() -> None:
    record = material()
    assert MaterialRecord.from_dict(record.to_dict()) == record
    assert record.property("thermal.value") == material_property()
    with pytest.raises(MaterialError, match="has no property"):
        record.property("missing")


def test_catalog_validation_and_lookup_failures() -> None:
    valid = catalog()
    assert MaterialCatalog.from_dict(valid.to_dict()) == valid
    assert valid.material("TEST MATERIAL").material_id == "test_material"
    assert valid.property("test_material", "thermal.value").evaluate() == 2.0

    with pytest.raises(MaterialError, match="no material selector"):
        valid.material("missing")
    with pytest.raises(MaterialError, match="no citation"):
        valid.citation("missing")
    with pytest.raises(MaterialError, match="unsupported"):
        replace(valid, schema_version="femx.material_catalog/v2")
    with pytest.raises(MaterialError, match="citation ids must be unique"):
        replace(valid, citations=(citation(), citation()))
    with pytest.raises(MaterialError, match="material ids must be unique"):
        replace(valid, materials=(material(), material()))
    with pytest.raises(MaterialError, match="shared"):
        replace(
            valid,
            materials=(material(), material("other", aliases=("Test material",))),
        )
    with pytest.raises(MaterialError, match="unknown sources"):
        replace(
            valid, materials=(material(properties=(material_property(citation_ids=("other",)),)),)
        )


def test_json_decoder_rejects_duplicate_nonfinite_invalid_and_nonobject_inputs() -> None:
    with pytest.raises(MaterialError, match="duplicate JSON key"):
        MaterialCatalog.from_json('{"schema_version":"a","schema_version":"b"}')
    with pytest.raises(MaterialError, match="non-finite JSON constant"):
        MaterialCatalog.from_json('{"value": NaN}')
    with pytest.raises(MaterialError, match="invalid material catalog JSON"):
        MaterialCatalog.from_json("{")
    with pytest.raises(MaterialError, match="must be an object"):
        MaterialCatalog.from_json("[]")
    with pytest.raises(MaterialError, match="invalid material catalog structure"):
        MaterialCatalog.from_json('{"schema_version":"femx.material_catalog/v1"}')


def test_decoded_shapes_and_optional_fields_are_validated() -> None:
    valid = catalog().to_dict()
    with pytest.raises(MaterialError, match="catalog citations must be an array"):
        MaterialCatalog.from_dict({**valid, "citations": {}})
    with pytest.raises(MaterialError, match="keys must be strings"):
        MaterialCatalog.from_dict({**valid, "citations": [{1: "bad"}]})

    source = citation().to_dict()
    source["authors"] = {}
    with pytest.raises(MaterialError, match="citation authors must be an array"):
        MaterialCitation.from_dict(source)
    source["authors"] = [1]
    with pytest.raises(MaterialError, match="must be a string"):
        MaterialCitation.from_dict(source)
    source = citation().to_dict()
    source["year"] = True
    with pytest.raises(MaterialError, match="year must be an integer"):
        MaterialCitation.from_dict(source)
    source = citation().to_dict()
    source["title"] = 1
    with pytest.raises(MaterialError, match="title must be a string"):
        MaterialCitation.from_dict(source)

    with pytest.raises(MaterialError, match="JSON number"):
        ClosedInterval.from_dict({"minimum": False, "maximum": 1.0})

    prop = material_property().to_dict()
    prop["validity"] = {"notes": []}
    decoded = MaterialProperty.from_dict(prop)
    assert decoded.validity == PropertyValidity()


def test_canonical_json_is_standard_json() -> None:
    encoded = builtin_catalog().canonical_json()
    decoded = json.loads(encoded)
    assert decoded["schema_version"] == "femx.material_catalog/v1"
    assert decoded["materials"][0]["id"] == "si_crystalline_intrinsic"
