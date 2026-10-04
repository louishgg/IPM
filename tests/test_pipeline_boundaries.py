"""Architectural guards for download, preparation, and analysis separation."""

import ast
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

import pytest

import backtest.analysis as analysis
from backtest.config import DEFAULT_CONFIG
from data_acquisition.provider_clients import (
    REQUIRED_CURL_CFFI_VERSION,
    REQUIRED_YFINANCE_VERSION,
)
from portfolio_core.strategies import build_registered_strategy


BACKTEST_PACKAGE = Path(__file__).resolve().parents[1] / "backtest"
LIVE_PACKAGE = Path(__file__).resolve().parents[1] / "live"
ROOT = Path(__file__).resolve().parents[1]
DATA_ACQUISITION_PACKAGE = ROOT / "data_acquisition"
PORTFOLIO_CORE_PACKAGE = ROOT / "portfolio_core"
NETWORK_MODULES = {"requests", "requests_cache", "yfinance"}
BINARY_STATE_MODULES = {"pickle", "sqlite3"}


def test_conda_environment_pins_required_yahoo_client_versions():
    environment = (ROOT / "environment.yml").read_text(encoding="utf-8")
    dependency_specs = {
        stripped[2:].partition("#")[0].strip()
        for line in environment.splitlines()
        if (stripped := line.strip()).startswith("- ")
    }

    assert {
        f"curl-cffi={REQUIRED_CURL_CFFI_VERSION}",
        f"yfinance={REQUIRED_YFINANCE_VERSION}",
    } <= dependency_specs


def _imports(path: Path) -> set[str]:
    return {
        module.split(".", 1)[0]
        for module in _imported_modules(path)
    }


def _relative_imports(path: Path) -> set[tuple[int, str]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        (node.level, node.module or "")
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level
    }


def _imported_modules(path: Path) -> set[str]:
    """Return complete absolute module names for architectural assertions."""
    tree = ast.parse(path.read_text(), filename=str(path))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_backtest_preparation_builder_has_no_strategy_or_analysis_dependencies():
    path = BACKTEST_PACKAGE / "preparation_builders.py"
    absolute_imports = _imported_modules(path)
    strategy_imports = {
        module
        for module in absolute_imports
        if module == "portfolio_core.strategies"
        or module.startswith("portfolio_core.strategies.")
    }
    forbidden_relative_imports = {
        (1, module)
        for module in ("analysis", "engine", "evaluation", "strategy", "strategies")
    }

    assert strategy_imports == set()
    assert not (_relative_imports(path) & forbidden_relative_imports)


def test_sector_acquisition_dependencies_are_one_way():
    artifacts = DATA_ACQUISITION_PACKAGE / "sector_acquisition_artifacts.py"
    planning = DATA_ACQUISITION_PACKAGE / "sector_acquisition_planning.py"
    execution = DATA_ACQUISITION_PACKAGE / "sector_acquisition_execution.py"

    artifact_imports = _imported_modules(artifacts)
    planning_imports = _imported_modules(planning)
    execution_imports = _imported_modules(execution)
    assert "data_acquisition.sector_acquisition_planning" not in artifact_imports
    assert "data_acquisition.sector_acquisition_execution" not in artifact_imports
    assert "data_acquisition.sector_acquisition_execution" not in planning_imports
    assert "data_acquisition.sector_acquisition_artifacts" in planning_imports
    assert "data_acquisition.sector_acquisition_artifacts" in execution_imports
    assert "data_acquisition.sector_acquisition_planning" in execution_imports


def test_sector_assignment_contracts_do_not_depend_on_resolution():
    assignments = PORTFOLIO_CORE_PACKAGE / "sector_assignments.py"
    resolution = PORTFOLIO_CORE_PACKAGE / "sector_resolution.py"
    assert "portfolio_core.sector_resolution" not in _imported_modules(assignments)
    assert "portfolio_core.sector_assignments" in _imported_modules(resolution)


def test_preparation_and_analysis_modules_do_not_import_provider_layers():
    paths = (
        BACKTEST_PACKAGE / "preparation_builders.py",
        BACKTEST_PACKAGE / "preparation_pipeline.py",
        BACKTEST_PACKAGE / "analysis.py",
        BACKTEST_PACKAGE / "evaluation.py",
        BACKTEST_PACKAGE / "brinson_attribution.py",
        LIVE_PACKAGE / "preparation_builders.py",
        LIVE_PACKAGE / "preparation_pipeline.py",
        LIVE_PACKAGE / "analysis.py",
        LIVE_PACKAGE / "analysis_data.py",
        LIVE_PACKAGE / "brinson_attribution.py",
    )
    for path in paths:
        imports = _imported_modules(path)
        assert not {
            module
            for module in imports
            if module.startswith("data_acquisition.provider")
            or module.startswith("data_acquisition.providers")
        }, path


def test_offline_imports_do_not_initialize_network_provider_modules():
    modules = (
        "portfolio_core.shares",
        "backtest.preparation_builders",
        "backtest.evaluation",
        "live.preparation_builders",
        "live.analysis_data",
    )
    script = (
        "import importlib, sys\n"
        f"modules = {modules!r}\n"
        "for name in modules:\n"
        "    importlib.import_module(name)\n"
        "forbidden = {'requests', 'requests_cache', 'yfinance', "
        "'data_acquisition.providers.sec', "
        "'data_acquisition.providers.wikipedia', "
        "'data_acquisition.providers.yahoo'}\n"
        "loaded = sorted(forbidden.intersection(sys.modules))\n"
        "raise SystemExit(','.join(loaded) if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_backtest_data_and_preparation_manifest_dependencies_are_acyclic():
    assert (1, "preparation_builders") not in _relative_imports(
        BACKTEST_PACKAGE / "data_loading.py"
    )
    assert (1, "data_loading") not in _relative_imports(
        BACKTEST_PACKAGE / "preparation_artifacts.py"
    )


def test_strategy_data_has_no_reverse_analysis_import():
    assert (1, "analysis") not in _relative_imports(
        LIVE_PACKAGE / "analysis_data.py"
    )


def test_share_planning_has_one_way_acquisition_dependency():
    plan_path = BACKTEST_PACKAGE / "acquisition_planning.py"
    preparation_path = BACKTEST_PACKAGE / "shares_preparation.py"
    plan_tree = ast.parse(plan_path.read_text(), filename=str(plan_path))
    preparation_tree = ast.parse(
        preparation_path.read_text(),
        filename=str(preparation_path),
    )

    def imports_relative(tree, module_name):
        return any(
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and (
                node.module == module_name
                or node.module is None
                and any(alias.name == module_name for alias in node.names)
            )
            for node in ast.walk(tree)
        )

    assert not imports_relative(plan_tree, "shares_preparation")
    assert imports_relative(preparation_tree, "acquisition_planning")


def test_acquisition_workflows_depend_on_plans_not_registration_facades():
    for package in (BACKTEST_PACKAGE, LIVE_PACKAGE):
        path = package / "acquisition_execution.py"
        imports = _relative_imports(path)
        assert (1, "acquisition_handlers") not in imports
        tree = ast.parse(path.read_text(), filename=str(path))
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.level == 1
            and (
                node.module == "acquisition_planning"
                or any(
                    alias.name == "acquisition_planning"
                    for alias in node.names
                )
            )
            for node in ast.walk(tree)
        )

    live_plan_imports = _relative_imports(LIVE_PACKAGE / "acquisition_planning.py")
    assert (1, "preparation_builders") not in live_plan_imports
    source_contract_imports = _relative_imports(
        LIVE_PACKAGE / "strategy_universe.py"
    )
    assert (1, "preparation_builders") not in source_contract_imports


def test_domain_packages_do_not_import_network_clients():
    for package in (BACKTEST_PACKAGE, LIVE_PACKAGE):
        network_importers = {
            path.relative_to(package).as_posix()
            for path in package.rglob("*.py")
            if _imports(path) & NETWORK_MODULES
        }
        assert network_importers == set()


def test_network_clients_are_centralized_in_data_acquisition():
    network_importers = {
        path.relative_to(DATA_ACQUISITION_PACKAGE).as_posix()
        for path in DATA_ACQUISITION_PACKAGE.rglob("*.py")
        if _imports(path) & NETWORK_MODULES
    }
    assert network_importers == {
        "provider_clients.py",
        "providers/wikipedia.py",
        "providers/yahoo.py",
    }


def test_portfolio_core_has_no_reverse_domain_dependency():
    reverse_importers = {
        path.relative_to(PORTFOLIO_CORE_PACKAGE).as_posix(): _imports(path)
        & {"backtest", "live"}
        for path in PORTFOLIO_CORE_PACKAGE.rglob("*.py")
        if _imports(path) & {"backtest", "live"}
    }
    assert reverse_importers == {}


def test_domain_packages_do_not_import_each_other():
    backtest_importers = {
        path.relative_to(BACKTEST_PACKAGE).as_posix()
        for path in BACKTEST_PACKAGE.rglob("*.py")
        if "live" in _imports(path)
    }
    live_importers = {
        path.relative_to(LIVE_PACKAGE).as_posix()
        for path in LIVE_PACKAGE.rglob("*.py")
        if "backtest" in _imports(path)
    }
    assert backtest_importers == set()
    assert live_importers == set()


def test_production_workflows_use_only_the_shared_policy_constant():
    workflows = (
        BACKTEST_PACKAGE / "acquisition_execution.py",
        LIVE_PACKAGE / "acquisition_execution.py",
    )
    for path in workflows:
        tree = ast.parse(path.read_text(), filename=str(path))
        policy_constructors = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "AcquisitionPolicy"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "AcquisitionPolicy"
            )
        ]
        assert policy_constructors == [], path

        engines = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name)
                and node.func.id == "SerialAcquisitionEngine"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "SerialAcquisitionEngine"
            )
        ]
        assert engines, path
        for call in engines:
            policy = next(
                (keyword.value for keyword in call.keywords if keyword.arg == "policy"),
                None,
            )
            assert isinstance(policy, ast.Name), (path, ast.dump(call))
            assert policy.id == "PRODUCTION_ACQUISITION_POLICY"


@pytest.mark.parametrize(
    "package",
    (BACKTEST_PACKAGE, LIVE_PACKAGE),
    ids=("backtest", "live"),
)
def test_domain_source_has_no_pickle_or_direct_sqlite_dependencies(package):
    for path in package.glob("*.py"):
        assert not (_imports(path) & BINARY_STATE_MODULES), path

        tree = ast.parse(path.read_text(), filename=str(path))
        forbidden_calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"read_pickle", "to_pickle"}
        }
        assert not forbidden_calls, (path, forbidden_calls)


def test_analysis_preflight_fails_before_any_output(monkeypatch):
    writes = []
    monkeypatch.setattr(analysis, "apply_runtime_settings", lambda: None)
    monkeypatch.setattr(
        analysis,
        "load_backtest_data",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        analysis,
        "load_benchmark",
        lambda *args: (_ for _ in ()).throw(RuntimeError("missing benchmark")),
    )
    for name in (
        "calculate_strategy_artifacts",
        "save_strategy_artifacts",
        "run_brinson_attribution",
        "save_brinson_result",
    ):
        monkeypatch.setattr(
            analysis,
            name,
            lambda *args, _name=name, **kwargs: writes.append(_name),
        )

    with pytest.raises(RuntimeError, match="missing benchmark"):
        strategy = build_registered_strategy("momentum")
        analysis.run_analysis(
            "strategy",
            config=replace(
                DEFAULT_CONFIG,
                strategy=strategy,
                paths=DEFAULT_CONFIG.paths.for_strategy(
                    strategy.strategy_id
                ),
            ),
        )

    assert writes == []
