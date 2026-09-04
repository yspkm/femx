import pytest
from tests.support import FakeArray

from femx.core.errors import ContractError
from femx.core.parameters import ParameterRole, ParameterSchema, ParameterSpec

pytestmark = pytest.mark.unit


def test_schema_binds_exact_scalar_and_array_values() -> None:
    schema = ParameterSchema(
        (
            ParameterSpec("conductivity", unit="W/(m*K)", lower_bound=0.0),
            ParameterSpec("density", unit="1", shape=(2, 3), role=ParameterRole.DESIGN),
        )
    )
    values = schema.bind({"conductivity": 2.5, "density": FakeArray((2, 3))})

    assert values["conductivity"] == 2.5
    assert schema.names == ("conductivity", "density")
    with pytest.raises(TypeError):
        values.values["new"] = 1.0  # type: ignore[index]


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"conductivity": -1.0}, "below"),
        ({"conductivity": FakeArray((1,))}, "cannot receive an array"),
    ],
)
def test_scalar_parameter_validation(values: dict[str, object], message: str) -> None:
    schema = ParameterSchema((ParameterSpec("conductivity", lower_bound=0.0),))
    with pytest.raises(ContractError, match=message):
        schema.bind(values)  # type: ignore[arg-type]


def test_schema_rejects_key_and_shape_mismatches() -> None:
    schema = ParameterSchema((ParameterSpec("field", shape=(2,)),))
    with pytest.raises(ContractError, match="key mismatch"):
        schema.bind({})
    with pytest.raises(ContractError, match="expected shape"):
        schema.bind({"field": FakeArray((3,))})


def test_parameter_schema_rejects_duplicates_and_invalid_bounds() -> None:
    with pytest.raises(ContractError, match="unique"):
        ParameterSchema((ParameterSpec("x"), ParameterSpec("x")))
    with pytest.raises(ContractError, match="reversed"):
        ParameterSpec("x", lower_bound=2.0, upper_bound=1.0)
    with pytest.raises(ContractError, match="complex"):
        ParameterSpec("x", lower_bound=0.0).validate_value(1 + 2j)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " x"}, "name"),
        ({"name": "x", "unit": ""}, "unit"),
        ({"name": "x", "shape": (0,)}, "invalid shape"),
    ],
)
def test_parameter_spec_rejects_ambiguous_schema(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ContractError, match=message):
        ParameterSpec(**kwargs)  # type: ignore[arg-type]


def test_array_and_upper_bound_branches_are_enforced() -> None:
    array_spec = ParameterSpec("array", shape=(2,))
    with pytest.raises(ContractError, match="requires array"):
        array_spec.validate_value(1.0)
    with pytest.raises(ContractError, match="above"):
        ParameterSpec("x", upper_bound=1.0).validate_value(2.0)
    ParameterSpec("phase").validate_value(1 + 2j)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "finite"),
        (float("inf"), "finite"),
        (complex(1.0, float("inf")), "finite"),
        (True, "not boolean"),
    ],
)
def test_parameter_values_must_be_finite_real_numbers(
    value: float | complex | bool, message: str
) -> None:
    with pytest.raises(ContractError, match=message):
        ParameterSpec("finite").validate_value(value)
