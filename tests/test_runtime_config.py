"""Tests for deterministic runtime setup."""

import importlib
import os
from pathlib import Path
import sys
import warnings


def test_runtime_settings_are_applied_only_when_called(monkeypatch, tmp_path):
    module_name = "portfolio_core.runtime_config"
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        filters_before_import = list(warnings.filters)

        runtime_config = importlib.import_module(module_name)

        assert "MPLCONFIGDIR" not in os.environ
        assert warnings.filters == filters_before_import

        monkeypatch.setattr(
            runtime_config.tempfile,
            "gettempdir",
            lambda: str(tmp_path),
        )
        runtime_config.apply_runtime_settings()
        warnings.warn("Timestamp.utcnow is deprecated", FutureWarning)
        warnings.warn("unrelated runtime warning", FutureWarning)

    expected_dir = tmp_path / "matplotlib"
    assert expected_dir.is_dir()
    assert Path(os.environ["MPLCONFIGDIR"]) == expected_dir
    assert [str(item.message) for item in caught] == [
        "unrelated runtime warning"
    ]
    assert caught[0].category is FutureWarning
