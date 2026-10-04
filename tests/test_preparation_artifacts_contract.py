"""Focused corruption tests for the unified schema-v7 preparation manifest."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from backtest.paths import BacktestPaths
from backtest.preparation_artifacts import _artifact_specs
from live.paths import LivePaths
from live.preparation_artifacts import (
    artifact_catalog as live_artifact_catalog,
    required_artifact_names,
    validate_preparation_manifest as validate_live_manifest,
    write_preparation_manifest as write_live_manifest,
)
from live.analysis_data import validate_analysis_manifest
from portfolio_core.artifacts import (
    PREPARATION_SCHEMA_VERSION,
    ArtifactSpec,
    artifact_shape,
    file_sha256,
    validate_preparation_manifest,
    write_preparation_manifest,
)


def test_file_sha256_matches_known_digest(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"abc")

    assert file_sha256(artifact) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )


def test_artifact_shape_uses_type_stable_bounded_csv_reads(
    tmp_path,
    monkeypatch,
):
    artifact = tmp_path / "mixed.csv"
    artifact.write_text(
        "Sparse,Value\n,1\n2026-01-31,2\n",
        encoding="utf-8",
    )
    calls = []
    original_read_csv = pd.read_csv

    def tracked_read_csv(*args, **kwargs):
        assert kwargs.get("dtype") in (str, "str", "string", object)
        # Permit a header read, chunked iteration, or another bounded read.
        nrows, chunksize = kwargs.get("nrows"), kwargs.get("chunksize")
        assert (
            (nrows is not None and nrows >= 0)
            or (chunksize is not None and chunksize > 0)
        ), "Artifact inspection must not load an entire CSV into memory"
        calls.append(dict(kwargs))
        return original_read_csv(*args, **kwargs)

    monkeypatch.setattr(
        "portfolio_core.artifacts.pd.read_csv",
        tracked_read_csv,
    )

    assert artifact_shape(ArtifactSpec("raw", "mixed", artifact)) == (2, 2)
    assert calls


def test_both_preparation_catalogs_depend_only_on_sector_evidence(tmp_path):
    expected = {"sector_history_snapshots", "sector_history_notices"}
    backtest_paths = BacktestPaths(tmp_path / "backtest")
    backtest_names = {
        spec.artifact
        for spec in _artifact_specs(
            backtest_paths,
            include_brinson=False,
        )
        if spec.artifact.startswith("sector_history_")
    }
    live_names = {
        spec.artifact
        for spec in live_artifact_catalog(LivePaths(tmp_path / "live"))
        if spec.artifact.startswith("sector_history_")
    }

    assert backtest_names == live_names == expected


def _shared_contract(tmp_path):
    raw = tmp_path / "raw.csv"
    core = tmp_path / "core.csv"
    later = tmp_path / "later.csv"
    raw.write_text("Value\nraw\n", encoding="utf-8")
    core.write_text("Value\n1\n", encoding="utf-8")
    later.write_text("Value\n2\n", encoding="utf-8")
    specs = (
        ArtifactSpec("raw", "raw_input", raw),
        ArtifactSpec("prepared", "core_output", core),
        ArtifactSpec("prepared", "later_output", later),
    )
    manifest = tmp_path / "preparation_manifest.csv"
    write_preparation_manifest(
        specs,
        manifest_path=manifest,
        repository_root=tmp_path,
    )
    return specs, manifest, core


def _validate_shared(
    specs,
    manifest,
    root,
    required=("core_output",),
    validate_raw_artifacts=(),
):
    validate_preparation_manifest(
        specs,
        manifest_path=manifest,
        repository_root=root,
        recovery_command="python -m example.prepare",
        required_artifacts=required,
        validate_raw_artifacts=validate_raw_artifacts,
    )


def test_manifest_accepts_known_later_rows_and_optional_absence(tmp_path):
    specs, manifest, _ = _shared_contract(tmp_path)
    _validate_shared(specs, manifest, tmp_path)

    frame = pd.read_csv(manifest)
    frame.loc[frame["Artifact"].ne("later_output")].to_csv(
        manifest,
        index=False,
    )
    _validate_shared(specs, manifest, tmp_path)

    with pytest.raises(RuntimeError, match="missing required artifacts"):
        _validate_shared(
            specs,
            manifest,
            tmp_path,
            required=("core_output", "later_output"),
        )


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("Schema_Version", PREPARATION_SCHEMA_VERSION - 1),
        ("Stage", "raw"),
        ("Relative_Path", "relocated/core.csv"),
        ("SHA256", "0" * 64),
        ("Rows", 99),
        ("Columns", 99),
        ("Rows", "not-a-number"),
        ("Columns", 1.5),
    ],
)
def test_manifest_rejects_corruption_in_every_recorded_field(
    tmp_path,
    column,
    replacement,
):
    specs, manifest, _ = _shared_contract(tmp_path)
    frame = pd.read_csv(manifest)
    if column in {"Rows", "Columns"} and not isinstance(replacement, int):
        frame[column] = frame[column].astype(object)
    frame.loc[frame["Artifact"].eq("core_output"), column] = replacement
    frame.to_csv(manifest, index=False)

    with pytest.raises(RuntimeError):
        _validate_shared(specs, manifest, tmp_path)


@pytest.mark.parametrize("malformation", ["missing_column", "extra_column"])
def test_manifest_requires_the_exact_seven_column_schema(
    tmp_path,
    malformation,
):
    specs, manifest, _ = _shared_contract(tmp_path)
    frame = pd.read_csv(manifest)
    if malformation == "missing_column":
        frame = frame.drop(columns="Columns")
    else:
        frame["Unexpected"] = "value"
    frame.to_csv(manifest, index=False)

    with pytest.raises(RuntimeError, match="Invalid preparation manifest schema"):
        _validate_shared(specs, manifest, tmp_path)


@pytest.mark.parametrize("corruption", ["unknown", "missing", "duplicate"])
def test_manifest_rejects_unknown_missing_and_duplicate_artifacts(
    tmp_path,
    corruption,
):
    specs, manifest, _ = _shared_contract(tmp_path)
    frame = pd.read_csv(manifest)
    core_row = frame.loc[frame["Artifact"].eq("core_output")]
    if corruption == "unknown":
        frame.loc[frame["Artifact"].eq("core_output"), "Artifact"] = "unknown"
    elif corruption == "missing":
        frame = frame.loc[frame["Artifact"].ne("core_output")]
    else:
        frame = pd.concat([frame, core_row], ignore_index=True)
    frame.to_csv(manifest, index=False)

    with pytest.raises(RuntimeError):
        _validate_shared(specs, manifest, tmp_path)


def test_manifest_rejects_stale_prepared_bytes(tmp_path):
    specs, manifest, core = _shared_contract(tmp_path)
    core.write_text("Value\nchanged\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or modified"):
        _validate_shared(specs, manifest, tmp_path)


def test_manifest_validates_only_explicitly_consumed_raw_bytes(tmp_path):
    specs, manifest, _ = _shared_contract(tmp_path)
    raw = specs[0].path
    raw.write_text("Value\nchanged\n", encoding="utf-8")

    _validate_shared(specs, manifest, tmp_path)
    with pytest.raises(RuntimeError, match="stale or modified"):
        _validate_shared(
            specs,
            manifest,
            tmp_path,
            validate_raw_artifacts={"raw_input"},
        )


def _write_complete_live_manifest(paths: LivePaths) -> dict[str, Path]:
    specs = live_artifact_catalog(paths)
    for spec in specs:
        spec.path.parent.mkdir(parents=True, exist_ok=True)
        spec.path.write_text(
            f"Value\n{spec.artifact}\n",
            encoding="utf-8",
        )
    write_live_manifest(paths, include_brinson=True)
    return {spec.artifact: spec.path for spec in specs}


@pytest.mark.parametrize(
    "artifact_name",
    ("shares_prepared", "benchmark_prepared"),
)
def test_final_live_brinson_manifest_rejects_modified_artifacts(
    tmp_path,
    artifact_name,
):
    paths = LivePaths(tmp_path / "live")
    artifact_paths = _write_complete_live_manifest(paths)
    artifact_path = artifact_paths[artifact_name]
    artifact_path.write_text("Value\nmodified\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        validate_live_manifest(
            paths,
            include_brinson=True,
            check_raw=True,
        )

    assert str(exc_info.value) == (
        f"Prepared pipeline artifact is stale or modified: {artifact_path}. "
        "Run `python -m live.prepare all` first."
    )


def _write_strategy_manifest(paths: LivePaths) -> pd.DataFrame:
    artifact = paths.market_daily_csv
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("Value\n1\n", encoding="utf-8")
    paths.raw_price_supplemental_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_price_supplemental_csv.write_text(
        "Date,Asset_ID,Provider_Symbol,Open,Close,Volume\n",
        encoding="utf-8",
    )
    paths.raw_price_supplemental_manifest_csv.write_text(
        "Schema_Version,Artifact\n2,yahoo_ohlcv\n",
        encoding="utf-8",
    )
    selected = tuple(
        spec
        for spec in live_artifact_catalog(paths)
        if spec.artifact
        in {
            "market_daily",
            "price_yahoo_supplement",
            "price_yahoo_supplement_manifest",
        }
    )
    return write_preparation_manifest(
        selected,
        manifest_path=paths.preparation_manifest_csv,
        repository_root=paths.project_root,
    )


def test_strategy_adapter_rejects_stale_prepared_bytes(tmp_path):
    paths = LivePaths(tmp_path / "live")
    _write_strategy_manifest(paths)
    validate_analysis_manifest(paths, {"market_daily"})

    paths.market_daily_csv.write_text("Value\nchanged\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="stale or modified"):
        validate_analysis_manifest(paths, {"market_daily"})


def test_live_required_artifacts_include_actions_and_supplement_provenance(tmp_path):
    paths = LivePaths(tmp_path / "live")
    mandatory = required_artifact_names(paths, include_brinson=False)
    assert {
        "corporate_action_policy",
        "corporate_action_policy_manifest",
        "corporate_action_events_prepared",
        "corporate_action_legs_prepared",
        "corporate_action_sources_prepared",
        "corporate_action_policy_prepared",
        "price_yahoo_supplement",
        "price_yahoo_supplement_manifest",
    } <= mandatory


def test_analysis_accepts_missing_raw_supplement_files_with_manifest_rows(tmp_path):
    paths = LivePaths(tmp_path / "live")
    _write_strategy_manifest(paths)
    paths.raw_price_supplemental_csv.unlink()
    paths.raw_price_supplemental_manifest_csv.unlink()

    validate_analysis_manifest(paths, {"market_daily"})


@pytest.mark.parametrize(
    "path_attribute",
    ("raw_price_supplemental_csv", "raw_price_supplemental_manifest_csv"),
    ids=(
        "price_yahoo_supplement-raw_price_supplemental_csv",
        "price_yahoo_supplement_manifest-raw_price_supplemental_manifest_csv",
    ),
)
def test_raw_supplement_files_are_checked_only_when_requested(
    tmp_path,
    path_attribute,
):
    paths = LivePaths(tmp_path / "live")
    _write_strategy_manifest(paths)
    missing_path = getattr(paths, path_attribute)
    missing_path.unlink()

    validate_live_manifest(
        paths,
        check_raw=False,
        required_artifacts={"market_daily"},
    )

    with pytest.raises(RuntimeError) as exc_info:
        validate_live_manifest(
            paths,
            check_raw=True,
            required_artifacts={"market_daily"},
        )

    message = str(exc_info.value)
    assert f"Prepared pipeline artifact is missing: {missing_path}" in message
    assert "Run `python -m live.prepare all` first." in message


@pytest.mark.parametrize(
    "artifact_name",
    ("price_yahoo_supplement", "price_yahoo_supplement_manifest"),
    ids=(
        "price_yahoo_supplement-raw_price_supplemental_csv",
        "price_yahoo_supplement_manifest-raw_price_supplemental_manifest_csv",
    ),
)
def test_analysis_requires_each_supplement_provenance_row(
    tmp_path,
    artifact_name,
):
    paths = LivePaths(tmp_path / "live")
    frame = _write_strategy_manifest(paths)
    frame.loc[frame["Artifact"].ne(artifact_name)].to_csv(
        paths.preparation_manifest_csv,
        index=False,
    )

    with pytest.raises(RuntimeError) as exc_info:
        validate_analysis_manifest(paths, {"market_daily"})

    message = str(exc_info.value)
    assert "missing required artifacts" in message
    assert artifact_name in message


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("Stage", "prepared"),
        ("Relative_Path", "relocated/yahoo_ohlcv.csv"),
    ],
)
def test_analysis_validates_supplement_row_identity_without_raw_files(
    tmp_path,
    column,
    replacement,
):
    paths = LivePaths(tmp_path / "live")
    frame = _write_strategy_manifest(paths)
    frame.loc[
        frame["Artifact"].eq("price_yahoo_supplement"), column
    ] = replacement
    frame.to_csv(paths.preparation_manifest_csv, index=False)
    paths.raw_price_supplemental_csv.unlink()
    paths.raw_price_supplemental_manifest_csv.unlink()

    with pytest.raises(RuntimeError, match="stale or relocated"):
        validate_analysis_manifest(paths, {"market_daily"})


@pytest.mark.parametrize(
    "path_attribute",
    (
        "raw_price_supplemental_csv",
        "raw_price_supplemental_manifest_csv",
    ),
    ids=("price_yahoo_supplement", "price_yahoo_supplement_manifest"),
)
def test_live_manifest_writer_requires_each_raw_supplement_file(
    tmp_path,
    path_attribute,
):
    paths = LivePaths(tmp_path / "live")
    _write_strategy_manifest(paths)
    getattr(paths, path_attribute).unlink()

    with pytest.raises(RuntimeError, match="authoritative live Yahoo supplement"):
        write_live_manifest(paths, include_brinson=False)
