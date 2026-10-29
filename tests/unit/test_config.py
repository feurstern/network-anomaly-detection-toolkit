from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from network_anomaly.config import StrictModel, load_yaml_config, parse_config
from network_anomaly.errors import ConfigError, ExitCode


class _Window(StrictModel):
    size: int = Field(default=12, gt=0)


class _Example(StrictModel):
    name: str
    window: _Window = _Window()
    ratio: float = Field(default=0.5, ge=0, le=1)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_yaml_is_parsed_into_the_model(tmp_path: Path) -> None:
    config = load_yaml_config(_write(tmp_path, "name: demo\nwindow:\n  size: 6\n"), _Example)
    assert config == _Example(name="demo", window=_Window(size=6))


def test_unknown_key_is_rejected_with_its_full_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_yaml_config(_write(tmp_path, "name: demo\nwindow:\n  sise: 6\n"), _Example)
    message = str(info.value)
    assert "config.yaml" in message
    assert "window.sise: unknown option" in message


def test_out_of_range_value_names_the_key_constraint_and_input(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_yaml_config(_write(tmp_path, "name: demo\nratio: 1.5\n"), _Example)
    message = str(info.value)
    assert "ratio: Input should be less than or equal to 1" in message
    assert "(got 1.5)" in message


def test_every_problem_is_reported_at_once(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as info:
        load_yaml_config(_write(tmp_path, "ratio: -1\nwindow:\n  size: 0\n"), _Example)
    message = str(info.value)
    assert "3 problem(s)" in message
    assert "name: Field required" in message
    assert "window.size" in message


def test_invalid_yaml_reports_the_line(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"invalid YAML at line \d+"):
        load_yaml_config(_write(tmp_path, "name: [unclosed\nratio: 0.1\n"), _Example)


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_yaml_config(_write(tmp_path, "- a\n- b\n"), _Example)


def test_missing_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="configuration file not found"):
        load_yaml_config(tmp_path / "absent.yaml", _Example)


def test_empty_file_is_validated_like_an_empty_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="name: Field required"):
        load_yaml_config(_write(tmp_path, ""), _Example)


def test_parsed_configs_are_immutable() -> None:
    config = parse_config({"name": "demo"}, _Example)
    with pytest.raises(ValidationError):
        config.name = "other"  # type: ignore[misc]


def test_config_errors_exit_with_code_4() -> None:
    assert ConfigError.exit_code is ExitCode.CONFIG
    assert int(ExitCode.CONFIG) == 4
