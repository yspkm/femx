import json
import runpy
import sys

import pytest

from femx import cli

pytestmark = pytest.mark.unit


def test_doctor_json_is_read_only_and_machine_readable(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(cli.ElmerInstallation, "discover", lambda _name="ElmerSolver": None)

    return_code = cli.main(["doctor", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert return_code == 0
    assert payload[0]["name"] == "python"
    assert {item["name"] for item in payload} == {
        "python",
        "jax",
        "fdtdx",
        "h5py",
        "meshio",
        "elmer",
    }


def test_doctor_fails_only_for_explicitly_required_missing_component(capsys, monkeypatch) -> None:
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)
    monkeypatch.setattr(cli.ElmerInstallation, "discover", lambda _name="ElmerSolver": None)

    assert cli.main(["doctor", "--require", "jax"]) == 1
    output = capsys.readouterr().out
    assert "jax" in output
    assert "No accelerator was initialized" in output


def test_python_module_entrypoint_dispatches_to_cli(capsys, monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["femx", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("femx.__main__", run_name="__main__")

    assert exit_info.value.code == 0
    assert "femx" in capsys.readouterr().out
