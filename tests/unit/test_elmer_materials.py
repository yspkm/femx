import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from femx.materials import (
    ElmerLegacyMaterial,
    ElmerLegacyMaterialLibrary,
    ElmerLegacyParameter,
    MaterialError,
    load_elmer_material_library,
)

pytestmark = pytest.mark.unit

XML = b"""<!DOCTYPE egmaterials>
<materiallibrary>
  <material name="Aluminium (generic)">
    <parameter name="Density">2700.0</parameter>
    <parameter name="Heat conductivity">237.0</parameter>
  </material>
  <material name="Silicon (solid)">
    <parameter name="Heat conductivity">Variable Temperature; Real; 0 156; End</parameter>
    <parameter name="Electric conductivity">1.0e-3</parameter>
  </material>
</materiallibrary>
"""


def write_xml(tmp_path: Path, encoded: bytes = XML) -> Path:
    path = tmp_path / "egmaterials.xml"
    path.write_bytes(encoded)
    return path


def library(tmp_path: Path) -> ElmerLegacyMaterialLibrary:
    return load_elmer_material_library(
        write_xml(tmp_path),
        source_revision="4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
    )


def test_elmer_import_preserves_source_identity_raw_values_and_scalar_parsing(
    tmp_path: Path,
) -> None:
    path = write_xml(tmp_path)
    imported = load_elmer_material_library(
        path,
        source_revision="4f2d7e4b99f8f0dcf2f7ac579e056969373bf594",
        selected_names=("Silicon (solid)",),
    )

    assert imported.source_path == str(path.resolve())
    assert imported.source_sha256 == hashlib.sha256(XML).hexdigest()
    assert imported.usage_status == "legacy_unverified"
    assert len(imported.digest()) == 64
    assert imported.digest() == replace(imported, source_path=str(tmp_path / "moved.xml")).digest()
    silicon = imported.material("Silicon (solid)")
    assert silicon.parameter("Heat conductivity").raw_value.startswith("Variable Temperature")
    assert silicon.parameter("Heat conductivity").scalar_value is None
    assert silicon.parameter("Electric conductivity").scalar_value == 1.0e-3
    assert imported.to_dict()["schema_version"] == "femx.elmer_gui_materials/v1"
    assert '"source_path":' in imported.canonical_json()


def test_elmer_import_keeps_file_order_and_rejects_fuzzy_lookup(tmp_path: Path) -> None:
    imported = library(tmp_path)

    assert tuple(material.name for material in imported.materials) == (
        "Aluminium (generic)",
        "Silicon (solid)",
    )
    assert imported.material("Aluminium (generic)").parameter("Density").scalar_value == 2700.0
    with pytest.raises(MaterialError, match="no selected material"):
        imported.material("aluminium")
    with pytest.raises(MaterialError, match="has no parameter"):
        imported.material("Silicon (solid)").parameter("Density")


@pytest.mark.parametrize(
    ("encoded", "message"),
    [
        (b"<", "invalid Elmer material XML"),
        (b"<wrong/>", "expected materiallibrary root"),
        (b"<materiallibrary><other/></materiallibrary>", "unexpected Elmer"),
        (b"<materiallibrary><material/></materiallibrary>", "missing a name"),
        (
            b"<materiallibrary><material name='A'><parameter name='x'>1</parameter></material><material name='A'><parameter name='y'>2</parameter></material></materiallibrary>",
            "duplicate Elmer material",
        ),
        (
            b"<materiallibrary><material name='A'><other/></material></materiallibrary>",
            "unexpected element",
        ),
        (
            b"<materiallibrary><material name='A'><parameter>1</parameter></material></materiallibrary>",
            "incomplete parameter",
        ),
        (
            b"<materiallibrary><material name='A'><parameter name='x'/></material></materiallibrary>",
            "incomplete parameter",
        ),
        (
            b"<materiallibrary><material name='A'><parameter name='x'>1</parameter><parameter name='x'>2</parameter></material></materiallibrary>",
            "duplicate parameter",
        ),
        (b"<materiallibrary><material name='A'/></materiallibrary>", "at least one parameter"),
    ],
)
def test_elmer_import_rejects_malformed_or_ambiguous_xml(
    tmp_path: Path, encoded: bytes, message: str
) -> None:
    with pytest.raises(MaterialError, match=message):
        load_elmer_material_library(
            write_xml(tmp_path, encoded),
            source_revision="revision",
        )


def test_elmer_import_rejects_missing_file_and_invalid_selections(tmp_path: Path) -> None:
    with pytest.raises(MaterialError, match="cannot read Elmer material library"):
        load_elmer_material_library(tmp_path / "missing.xml", source_revision="revision")
    path = write_xml(tmp_path)
    with pytest.raises(MaterialError, match="non-empty and unique"):
        load_elmer_material_library(path, source_revision="revision", selected_names=())
    with pytest.raises(MaterialError, match="non-empty and unique"):
        load_elmer_material_library(
            path,
            source_revision="revision",
            selected_names=("Silicon (solid)", "Silicon (solid)"),
        )
    with pytest.raises(MaterialError, match="selection is missing"):
        load_elmer_material_library(
            path,
            source_revision="revision",
            selected_names=("Germanium",),
        )


def test_legacy_dataclasses_reject_invalid_direct_construction(tmp_path: Path) -> None:
    parameter = ElmerLegacyParameter("Density", "2700", 2700.0)
    material = ElmerLegacyMaterial("Aluminium", (parameter,))
    imported = ElmerLegacyMaterialLibrary(
        source_path=str((tmp_path / "source.xml").resolve()),
        source_sha256="0" * 64,
        source_revision="revision",
        materials=(material,),
    )

    assert parameter.to_dict()["scalar_value"] == 2700.0
    assert material.to_dict()["name"] == "Aluminium"
    with pytest.raises(MaterialError, match="parameter name"):
        replace(parameter, name="")
    with pytest.raises(MaterialError, match="parameter value"):
        replace(parameter, raw_value=" bad ")
    with pytest.raises(MaterialError, match="must be finite"):
        replace(parameter, scalar_value=float("inf"))
    with pytest.raises(MaterialError, match="material name"):
        replace(material, name="")
    with pytest.raises(MaterialError, match="at least one parameter"):
        replace(material, parameters=())
    with pytest.raises(MaterialError, match="duplicate parameters"):
        replace(material, parameters=(parameter, parameter))
    with pytest.raises(MaterialError, match="must be absolute"):
        replace(imported, source_path="relative.xml")
    with pytest.raises(MaterialError, match="lowercase SHA-256"):
        replace(imported, source_sha256="bad")
    with pytest.raises(MaterialError, match="revision"):
        replace(imported, source_revision="")
    with pytest.raises(MaterialError, match="legacy_unverified"):
        replace(imported, usage_status="trusted")
    with pytest.raises(MaterialError, match="unsupported"):
        replace(imported, schema_version="v2")
    with pytest.raises(MaterialError, match="selection cannot be empty"):
        replace(imported, materials=())
    with pytest.raises(MaterialError, match="names must be unique"):
        replace(imported, materials=(material, material))


def test_nonfinite_elmer_numeric_text_remains_raw_only(tmp_path: Path) -> None:
    encoded = b"<materiallibrary><material name='A'><parameter name='x'>nan</parameter></material></materiallibrary>"
    imported = load_elmer_material_library(write_xml(tmp_path, encoded), source_revision="revision")
    assert imported.material("A").parameter("x").scalar_value is None
