import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLE = Path(__file__).parents[2] / "examples" / "python_api.py"


def test_python_api_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    spec = importlib.util.spec_from_file_location("python_api_example", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(hosts=2, days=14, seed=3)
    output = capsys.readouterr().out
    assert "alerts in the test period" in output
    assert "Event level" in output
