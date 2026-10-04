"""Tests for sector evidence schemas and provenance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_acquisition.contracts import (
    AcquisitionStatus,
    write_acquisition_statuses,
)
from data_acquisition.sector_acquisition_planning import build_sector_acquisition_requests
from portfolio_core.sector_evidence import (
    NOTICE_COLUMNS,
    SectorHistoryPaths,
    load_sector_evidence,
    load_sector_notices,
    load_sector_snapshots,
    normalize_gics_sector,
    validate_sector_acquisition_manifest,
)
from _sector_test_helpers import (
    _acquisition_requirements,
    _notice,
    _notices,
    _snapshots,
    _write_sector_manifest,
)


@pytest.mark.parametrize(
    ("source_url", "message"),
    [
        (
            "https://example.test/w/index.php?"
            "title=List_of_S%26P_500_companies&oldid=101",
            "canonical HTTPS revision URLs",
        ),
        (
            "https://en.wikipedia.org/w/index.php?title=Wrong_Page&oldid=101",
            "title must identify",
        ),
        (
            "https://en.wikipedia.org/w/index.php?"
            "title=List_of_S%26P_500_companies",
            "oldid must exactly match",
        ),
        (
            "https://en.wikipedia.org/w/index.php?"
            "title=List_of_S%26P_500_companies&oldid=999",
            "oldid must exactly match",
        ),
        (
            "https://en.wikipedia.org/w/index.php?"
            "title=List_of_S%26P_500_companies&oldid=101&view=1",
            "only title and oldid",
        ),
    ],
)
def test_snapshot_loader_rejects_invalid_revision_urls(
    tmp_path: Path,
    source_url: str,
    message: str,
):
    frame = _snapshots([("2020-01-31", "AAA", "Industrials")])
    frame.loc[:, "Source_URL"] = source_url
    path = tmp_path / "snapshots.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_sector_snapshots(path)

def test_snapshot_loader_rejects_multiple_urls_for_one_date(tmp_path: Path):
    frame = _snapshots([
        ("2020-01-31", "AAA", "Industrials"),
        ("2020-01-31", "BBB", "Financials"),
    ])
    frame.loc[:, "Revision_ID"] = "101"
    frame.loc[:, "Revision_Timestamp_UTC"] = "2020-01-30T12:00:00Z"
    frame.loc[1, "Source_URL"] = (
        "https://en.wikipedia.org/w/index.php?"
        "oldid=101&title=List_of_S%26P_500_companies"
    )
    path = tmp_path / "snapshots.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError, match="exactly one revision"):
        load_sector_snapshots(path)

def test_notice_loader_rejects_nonofficial_source_host(tmp_path: Path):
    path = tmp_path / "notices.csv"
    pd.DataFrame([
        _notice(Source_URL="https://example.test/not-a-real-notice")
    ], columns=NOTICE_COLUMNS).to_csv(path, index=False)

    with pytest.raises(ValueError, match="approved official S&P Global"):
        load_sector_notices(path)

def test_preparation_evidence_validation_ignores_operational_status_bytes(
    tmp_path: Path,
):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    requirements = _acquisition_requirements()
    request = build_sector_acquisition_requests(requirements)[0]
    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(paths.notices_csv, index=False)
    write_acquisition_statuses(
        paths.acquisition_status_csv,
        (AcquisitionStatus.pending(request),),
    )
    _write_sector_manifest(paths, tmp_path)

    paths.acquisition_status_csv.write_bytes(
        paths.acquisition_status_csv.read_bytes() + b"\n"
    )

    assert len(
        load_sector_evidence(paths, repository_root=tmp_path).snapshots
    ) == 1
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_sector_acquisition_manifest(
            paths,
            repository_root=tmp_path,
        )

@pytest.mark.parametrize(
    "label",
    [
        "Communication Services",
        "Telecommunication Services",
        "Telecommunications Services",
    ],
)
def test_code_50_is_stable_across_historical_labels(label):
    assert normalize_gics_sector(label)[0] == "50"

def test_notice_loader_rejects_unreviewed_rows(tmp_path):
    path = tmp_path / "notices.csv"
    _notices([_notice(Review_Status="unreviewed")]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Unreviewed"):
        load_sector_notices(path)


def test_sector_evidence_requires_analytical_manifest(tmp_path):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(paths.notices_csv, index=False)

    with pytest.raises(ValueError, match="canonical acquisition artifact catalog"):
        load_sector_evidence(paths, repository_root=tmp_path)


@pytest.mark.parametrize("artifact_name", ("snapshots_csv", "notices_csv"))
def test_sector_evidence_rejects_drifted_analytical_artifacts(
    tmp_path,
    artifact_name,
):
    paths = SectorHistoryPaths.from_project_root(tmp_path)
    paths.directory.mkdir(parents=True)
    request = build_sector_acquisition_requests(_acquisition_requirements())[0]
    _snapshots([("2020-01-31", "AAA", "Energy")]).to_csv(
        paths.snapshots_csv,
        index=False,
    )
    _notices([_notice(Ticker="OTHER")]).to_csv(paths.notices_csv, index=False)
    write_acquisition_statuses(
        paths.acquisition_status_csv,
        (AcquisitionStatus.pending(request),),
    )
    _write_sector_manifest(paths, tmp_path)
    artifact = getattr(paths, artifact_name)
    artifact.write_bytes(artifact.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="manifest hash mismatch"):
        load_sector_evidence(paths, repository_root=tmp_path)

@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"Index_Name": "S&P MidCap 400"}, "unsupported indices"),
        ({"Action": "Replacement"}, "unsupported actions"),
    ],
)
def test_notice_loader_rejects_non_sp500_or_unknown_actions(
    tmp_path,
    change,
    message,
):
    path = tmp_path / "notices.csv"
    _notices([_notice(**change)]).to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        load_sector_notices(path)
