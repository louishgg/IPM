"""Canonical raw-share and reviewed preparation contracts."""
import pandas as pd
import pytest
from portfolio_core.shares import (RAW_SHARES_COLUMNS,
    select_canonical_share_observations_asof,
    load_raw_shares, validate_raw_shares)
RAW_COLUMNS = list(RAW_SHARES_COLUMNS)

def test_raw_loader_canonicalizes_and_preserves_duplicate_date_sequence(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame([
        {
            "Asset_ID": "BBB.K",
            "Provider_Symbol": "BBB",
            "Effective_Start": "",
            "Effective_End": "2022-12-31",
            "Date": "2022-02-01",
            "Observation_Sequence": 0,
            "Shares_Outstanding": 200.0,
        },
        {
            "Asset_ID": "AAA.O",
            "Provider_Symbol": "AAA",
            "Effective_Start": "",
            "Effective_End": "",
            "Date": "2022-01-31",
            "Observation_Sequence": 1,
            "Shares_Outstanding": 110.0,
        },
        {
            "Asset_ID": "AAA.O",
            "Provider_Symbol": "AAA",
            "Effective_Start": "",
            "Effective_End": "",
            "Date": "2022-01-31",
            "Observation_Sequence": 0,
            "Shares_Outstanding": 100.0,
        },
    ], columns=RAW_COLUMNS).to_csv(path, index=False)
    raw = load_raw_shares(path)

    assert tuple(raw.columns) == RAW_SHARES_COLUMNS
    assert raw[["Asset_ID", "Observation_Sequence"]].to_records(
        index=False
    ).tolist() == [("AAA.O", 0), ("AAA.O", 1), ("BBB.K", 0)]
    assert raw["Date"].tolist() == [
        pd.Timestamp("2022-01-31"),
        pd.Timestamp("2022-01-31"),
        pd.Timestamp("2022-02-01"),
    ]
    assert raw["Shares_Outstanding"].tolist() == [100.0, 110.0, 200.0]
    assert raw["Effective_Start"].tolist() == ["", "", ""]
    assert raw["Effective_End"].tolist() == [
        "",
        "",
        "2022-12-31",
    ]
    assert pd.api.types.is_datetime64_any_dtype(raw["Date"])
    assert raw["Observation_Sequence"].dtype == "int64"
    assert raw["Shares_Outstanding"].dtype == "float64"
    selected = select_canonical_share_observations_asof(
        raw, [("2022-01-31", "AAA.O")],
    )
    assert selected["Shares_Outstanding"].tolist() == [110.0]


def test_raw_share_loader_requires_exact_columns(tmp_path):
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        [["AAA", "AAA", "", "", "2022-01-31", 0, 100.0, "extra"]],
        columns=[*RAW_COLUMNS, "Extra"],
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Raw shares must contain exactly"):
        load_raw_shares(path)


@pytest.mark.parametrize(
    ("effective_start", "effective_end", "observation_date", "message"),
    [
        ("invalid", "", "2022-01-31", "invalid effective boundary"),
        ("2022-02-01", "", "2022-01-31", "outside their inclusive effective"),
    ],
)
def test_raw_share_loader_rejects_invalid_identity_intervals(
    tmp_path,
    effective_start,
    effective_end,
    observation_date,
    message,
):
    path = tmp_path / "raw.csv"
    pd.DataFrame(
        [[
            "AAA",
            "AAA",
            effective_start,
            effective_end,
            observation_date,
            0,
            100.0,
        ]],
        columns=RAW_COLUMNS,
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match=message):
        load_raw_shares(path)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Observation_Sequence", "invalid"),
        ("Shares_Outstanding", "invalid"),
    ],
)
def test_raw_share_loader_rejects_invalid_numeric_dtypes(tmp_path, column, value):
    path = tmp_path / "raw.csv"
    row = {
        "Asset_ID": "AAA",
        "Provider_Symbol": "AAA",
        "Effective_Start": "",
        "Effective_End": "",
        "Date": "2022-01-31",
        "Observation_Sequence": 0,
        "Shares_Outstanding": 100.0,
    }
    row[column] = value
    pd.DataFrame([row], columns=RAW_COLUMNS).to_csv(path, index=False)

    with pytest.raises(ValueError):
        load_raw_shares(path)


def test_effective_dated_fb_meta_and_viac_para_are_joined_without_time_travel():
    dates = pd.DatetimeIndex([
        "2022-01-31",
        "2022-02-28",
        "2022-05-31",
        "2022-06-30",
    ])
    raw = validate_raw_shares(pd.DataFrame([
        ["META.O", "FB", "", "2022-06-08", "2022-01-15", 0, 2_800.0],
        ["META.O", "META", "2022-06-09", "", "2022-06-09", 0, 2_700.0],
        ["PSKY.O", "VIAC", "", "2022-02-16", "2022-01-15", 0, 650.0],
        ["PSKY.O", "PARA", "2022-02-17", "", "2022-02-18", 0, 640.0],
    ], columns=RAW_COLUMNS))

    selected = select_canonical_share_observations_asof(
        raw, [(date, asset) for date in dates for asset in ("META.O", "PSKY.O")],
    )
    aligned = selected.pivot(index="Date", columns="Asset_ID", values="Shares_Outstanding")

    assert aligned["META.O"].tolist() == [2_800.0, 2_800.0, 2_800.0, 2_700.0]
    assert aligned["PSKY.O"].tolist() == [650.0, 640.0, 640.0, 640.0]


def test_effective_dated_new_symbol_never_fills_pre_rename_boundaries():
    raw = validate_raw_shares(pd.DataFrame([
        ["META.O", "META", "2022-06-09", "", "2022-06-09", 0, 2_700.0],
    ], columns=RAW_COLUMNS))
    dates = pd.DatetimeIndex(["2022-01-31", "2022-06-30"])
    selected = select_canonical_share_observations_asof(
        raw, [(date, "META.O") for date in dates],
    )
    aligned = selected.pivot(
        index="Date", columns="Asset_ID", values="Shares_Outstanding",
    ).reindex(dates)

    assert pd.isna(aligned.loc[pd.Timestamp("2022-01-31"), "META.O"])
    assert aligned.loc[pd.Timestamp("2022-06-30"), "META.O"] == 2_700.0


@pytest.mark.parametrize("sequences", [[0, 0], [0, 2], [1, 2]])
def test_raw_loader_rejects_invalid_observation_sequence(
    tmp_path, sequences
):
    path = tmp_path / "raw.csv"
    pd.DataFrame([
        {
            "Asset_ID": "AAA.O",
            "Provider_Symbol": "AAA",
            "Effective_Start": "",
            "Effective_End": "",
            "Date": "2022-01-31",
            "Observation_Sequence": sequence,
            "Shares_Outstanding": 100.0 + i,
        }
        for i, sequence in enumerate(sequences)
    ], columns=RAW_COLUMNS).to_csv(path, index=False)
    with pytest.raises(
        ValueError, match="[Oo]bservation[_ ]sequence|duplicate"
    ):
        load_raw_shares(path)
