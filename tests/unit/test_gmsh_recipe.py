import pytest

from femx.core.errors import ContractError
from femx.meshing.gmsh import RectangularWaveguideCrossSection

pytestmark = pytest.mark.unit


def test_waveguide_recipe_renders_si_units_and_stable_physical_groups() -> None:
    recipe = RectangularWaveguideCrossSection()
    rendered = recipe.render_geo()

    assert recipe.coordinate_scale_to_m == 1.0e-6
    assert 'Physical Surface("cladding", 101)' in rendered
    assert 'Physical Surface("core", 102)' in rendered
    assert 'Physical Curve("bottom", 201)' in rendered
    assert 'Physical Curve("right", 202)' in rendered
    assert 'Physical Curve("top", 203)' in rendered
    assert 'Physical Curve("left", 204)' in rendered
    assert "core_interface" not in rendered
    assert "Point(5) = {-0.25, -0.11, 0, lc_core};" in rendered
    assert rendered == RectangularWaveguideCrossSection().render_geo()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"core_width_m": 0.0}, "finite and positive"),
        ({"core_width_m": float("nan")}, "finite and positive"),
        ({"core_width_m": 4.0e-6}, "core width"),
        ({"core_height_m": 3.0e-6}, "core height"),
        ({"core_mesh_size_m": 0.5e-6}, "mesh size"),
    ],
)
def test_waveguide_recipe_rejects_invalid_geometry(kwargs, message: str) -> None:
    with pytest.raises(ContractError, match=message):
        RectangularWaveguideCrossSection(**kwargs)
