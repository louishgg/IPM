"""Reviewed monthly closes, separate from daily execution and liquidity data.

Preparation owns the source merge and dividend reconstruction. Consumers use
only its prepared tables; a monthly repair never creates a daily quote or volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_core.artifacts import ArtifactOrigin, file_sha256, validate_exact_manifest_catalog
from portfolio_core.provider_identity import BacktestProviderIdentityResolver
from portfolio_core.security_identity import load_security_identity_bundle


MONTHLY_COLUMNS = (
    "Month", "Asset_ID", "Observation_Date", "Close", "Volume", "Price_Source",
    "Nominal_Close", "Adjustment_Factor", "Evidence_ID", "Available_Date",
)
OBSERVATION_COLUMNS = (
    "Evidence_ID", "Asset_ID", "Observation_Date", "Provider", "Provider_Symbol",
    "Identity_Rule", "Mapping_ID", "Currency", "Nominal_Close", "Role",
    "Adjustment_Method", "Adjustment_Through", "Anchor_Date", "Anchor_Nominal_Close",
    "Source_File", "Source_SHA256", "Source_URL", "Captured_At_UTC",
    "Published_Date", "Review_Status", "Notes",
)
DIVIDEND_COLUMNS = (
    "Evidence_ID", "Asset_ID", "Ex_Date", "Dividend", "Previous_Session",
    "Previous_Close", "Previous_Close_Method", "Anchor_Date", "Anchor_Nominal_Close",
    "Currency", "Source_File", "Source_SHA256", "Source_URL", "Captured_At_UTC",
    "Published_Date", "Review_Status", "Notes",
)
PREPARED_DIVIDEND_COLUMNS = (*DIVIDEND_COLUMNS, "Factor", "Available_Date")
RECONSTRUCTED_SOURCE = "reuters_reconstructed"
OVERRIDE_SOURCE = "reviewed_monthly_correction"


def _dates(frame, columns):
    for column in columns:
        frame[column] = pd.to_datetime(frame[column].replace("", pd.NaT), errors="raise")
        if frame[column].dt.tz is not None or not frame[column].dropna().eq(
            frame[column].dropna().dt.normalize()
        ).all():
            raise ValueError(f"Monthly evidence requires naive date-only {column}")


def _positive(frame, columns, *, optional=()):
    for column in columns:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
        values = frame[column]
        valid = np.isfinite(values) & values.gt(0)
        if column in optional:
            valid |= values.isna()
        if not valid.all():
            raise ValueError(f"Monthly evidence contains invalid {column}")


def _source_rows(frame, root):
    """Check reproducibility inputs, without treating capture as publication."""
    if frame.Evidence_ID.eq("").any() or frame.Evidence_ID.duplicated().any():
        raise ValueError("Monthly evidence requires unique nonblank Evidence_IDs")
    if not frame.Review_Status.eq("approved").all() or not frame.Currency.eq("USD").all():
        raise ValueError("Monthly evidence must be reviewed USD observations")
    if frame.Asset_ID.eq("").any() or frame.Source_URL.eq("").any():
        raise ValueError("Monthly evidence requires security and source provenance")
    if pd.to_datetime(frame.Captured_At_UTC, utc=True, errors="raise").isna().any():
        raise ValueError("Monthly evidence requires capture timestamps")
    if not frame.Source_SHA256.str.fullmatch(r"[a-f0-9]{64}").all():
        raise ValueError("Monthly evidence requires source SHA256 hashes")
    for name, rows in frame.loc[frame.Source_File.ne("")].groupby("Source_File"):
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"Invalid monthly source file: {name}")
        if set(rows.Source_SHA256) != {file_sha256(path)}:
            raise ValueError(f"Monthly source changed; review its evidence again: {name}")


def _daily_close(market, asset, date):
    values = market.loc[market.Asset_ID.eq(asset) & market.Date.eq(date), "Close"]
    if len(values) != 1 or not np.isfinite(values.iloc[0]) or values.iloc[0] <= 0:
        raise ValueError(f"Missing exact adjusted anchor for {asset} at {date}")
    return float(values.iloc[0])


def load_monthly_evidence(paths, market):
    """Load the small reviewed repair ledger and verify canonical Reuters IDs."""
    directory = paths.raw_monthly_dir
    validate_exact_manifest_catalog(
        directory / "artifact_manifest.csv", scope="live", dataset="monthly_history",
        expected_origins={"observations.csv": ArtifactOrigin.MANUAL,
                          "dividends.csv": ArtifactOrigin.MANUAL,
                          "basis_checks.csv": ArtifactOrigin.MANUAL}, base_dir=directory,
    )
    observations = pd.read_csv(directory / "observations.csv", keep_default_na=False)
    dividends = pd.read_csv(directory / "dividends.csv", keep_default_na=False)
    for frame, columns in ((observations, OBSERVATION_COLUMNS), (dividends, DIVIDEND_COLUMNS)):
        if tuple(frame.columns) != columns:
            raise ValueError("Monthly repair evidence schema is inconsistent")
        _source_rows(frame, paths.project_root)
    _dates(observations, ("Observation_Date", "Adjustment_Through", "Anchor_Date", "Published_Date"))
    _positive(observations, ("Nominal_Close", "Anchor_Nominal_Close"), optional=("Anchor_Nominal_Close",))
    if observations.duplicated(["Asset_ID", "Observation_Date"]).any():
        raise ValueError("Monthly repair identities/dates must be unique")
    if not observations.Role.isin({"fallback", "override", "event_price"}).all():
        raise ValueError("Unknown monthly repair role")
    if not observations.Adjustment_Method.isin({"dividend_chain", "yahoo_anchor", "nominal_only"}).all():
        raise ValueError("Unknown monthly adjustment method")
    # Reuse the existing resolver; a blank Mapping_ID is valid for unique RICs.
    reuters_path = paths.project_root / "data/shared/supplied/reuters/SP500_Full_2014_2026_Cleaned.csv"
    reuters = pd.read_csv(reuters_path, usecols=["Date", "RIC", "Price Close"], parse_dates=["Date"])
    resolver = BacktestProviderIdentityResolver(
        load_security_identity_bundle(paths.project_root),
        pd.DataFrame({"Provider": "reuters", "Provider_Symbol": reuters.RIC.unique()}),
    )
    for row in observations.itertuples():
        if row.Provider == "reuters":
            identity = resolver.resolve(row.Asset_ID, row.Observation_Date)
            mappings = ";".join(i.mapping_id for i in identity.identities)
            if (row.Provider_Symbol not in identity.provider_symbols or
                    row.Identity_Rule != identity.resolution_method or row.Mapping_ID != mappings):
                raise ValueError(f"Monthly identity conflict for {row.Evidence_ID}")
            quotes = reuters.loc[reuters.RIC.eq(row.Provider_Symbol)
                                 & reuters.Date.eq(row.Observation_Date), "Price Close"]
            if len(quotes) != 1 or not np.isclose(quotes.iloc[0], row.Nominal_Close, rtol=0, atol=1e-10):
                raise ValueError(f"Reuters source quote disagrees for {row.Evidence_ID}")
        elif row.Provider == "yahoo_supplement" and row.Role == "event_price":
            if row.Identity_Rule != "reviewed_no_dividend_nominal_equivalence":
                raise ValueError(f"Unreviewed nominal equivalence for {row.Evidence_ID}")
            quotes = market.loc[market.Asset_ID.eq(row.Asset_ID) & market.Date.eq(row.Observation_Date)]
            if (len(quotes) != 1 or quotes.iloc[0].Yahoo_Ticker != row.Provider_Symbol or
                    not np.isclose(quotes.iloc[0].Close, row.Nominal_Close, atol=1e-10, rtol=0)):
                raise ValueError(f"Yahoo nominal evidence disagrees for {row.Evidence_ID}")
        else:
            raise ValueError(f"Unsupported monthly provider for {row.Evidence_ID}")

    _dates(dividends, ("Ex_Date", "Previous_Session", "Anchor_Date", "Published_Date"))
    _positive(dividends, ("Dividend", "Previous_Close", "Anchor_Nominal_Close"),
              optional=("Anchor_Nominal_Close",))
    for row in dividends.itertuples():
        if row.Previous_Close_Method == "cache_implied":
            # P_pre = adjusted_pre / (adjusted_post / nominal_post) + D.
            # This reconstructs the pre-ex quote using one cache vintage; the
            # monthly join is independently anchored to the reviewed later block.
            post_factor = _daily_close(market, row.Asset_ID, row.Anchor_Date) / row.Anchor_Nominal_Close
            inferred = _daily_close(market, row.Asset_ID, row.Previous_Session) / post_factor + row.Dividend
            if not np.isclose(inferred, row.Previous_Close, rtol=0, atol=1e-7):
                raise ValueError(f"Cached dividend bridge changed: {row.Evidence_ID}")
        elif row.Previous_Close_Method != "observed":
            raise ValueError(f"Unsupported preceding-close method: {row.Evidence_ID}")
    dividends["Factor"] = 1 - dividends.Dividend / dividends.Previous_Close
    dividends["Available_Date"] = dividends[["Ex_Date", "Published_Date"]].max(axis=1)
    return observations, validate_dividends(dividends)


def validate_dividends(frame):
    result = frame.copy()
    if tuple(result.columns) != PREPARED_DIVIDEND_COLUMNS:
        raise ValueError("Prepared monthly dividend schema is inconsistent")
    _dates(result, ("Ex_Date", "Previous_Session", "Anchor_Date", "Published_Date", "Available_Date"))
    _positive(result, ("Dividend", "Previous_Close", "Factor"))
    if (result.duplicated(["Asset_ID", "Ex_Date"]).any() or
            not result.Previous_Session.lt(result.Ex_Date).all() or
            not result.Factor.lt(1).all() or not result.Available_Date.ge(result.Ex_Date).all() or
            not np.allclose(result.Factor, 1 - result.Dividend / result.Previous_Close, rtol=0, atol=1e-12)):
        raise ValueError("Invalid or duplicate monthly dividend adjustment")
    return result


def empty_dividends():
    return pd.DataFrame(columns=PREPARED_DIVIDEND_COLUMNS)


def aggregate_monthly_market(market, *, through):
    """Aggregate real observations, retaining the date of each closing quote.

    A stale last quote remains inspectable but is not a month-end close. Volume
    is summed from actual daily rows with min_count=1, never filled with zero.
    """
    rows = market.loc[market.Date.le(pd.Timestamp(through))].copy()
    rows["Month"] = rows.Date + pd.offsets.MonthEnd(0)
    # The caller validates that its final signal cutoff is a completed session.
    rows = rows.loc[rows.Month.le(pd.Timestamp(through) + pd.offsets.MonthEnd(0))]
    dates = rows.groupby("Month").Date.max()
    valid = rows.loc[np.isfinite(rows.Close) & rows.Close.gt(0)].sort_values(["Date", "Asset_ID"])
    monthly = valid.drop_duplicates(["Month", "Asset_ID"], keep="last").rename(columns={"Date": "Observation_Date"})
    volume = rows.groupby(["Month", "Asset_ID"]).Volume.sum(min_count=1)
    monthly["Volume"] = [volume.get((r.Month, r.Asset_ID), np.nan) for r in monthly.itertuples()]
    monthly.loc[monthly.Observation_Date.ne(monthly.Month.map(dates)), "Close"] = np.nan
    monthly["Nominal_Close"] = np.nan
    monthly["Adjustment_Factor"] = np.nan
    monthly["Evidence_ID"] = ""
    monthly["Available_Date"] = monthly.Observation_Date
    return monthly.loc[:, MONTHLY_COLUMNS].sort_values(["Month", "Asset_ID"]).reset_index(drop=True)


def compose_monthly_history(market, observations, dividends, *, through):
    """Prefer observed Yahoo closes except explicit, reviewed basis corrections."""
    monthly = aggregate_monthly_market(market, through=through).set_index(["Month", "Asset_ID"])
    daily = market.loc[market.Date.le(pd.Timestamp(through))]
    calendar = daily.groupby(daily.Date + pd.offsets.MonthEnd(0)).Date.max()
    for row in observations.itertuples():
        label = row.Observation_Date + pd.offsets.MonthEnd(0)
        if row.Observation_Date > pd.Timestamp(through):
            continue
        if row.Observation_Date != calendar.get(label):
            raise ValueError(f"Repair is not the exact month-end observation: {row.Evidence_ID}")
        key = (label, row.Asset_ID)
        if key not in monthly.index:
            monthly.loc[key, :] = [row.Observation_Date, np.nan, np.nan, "", np.nan, np.nan, "", row.Observation_Date]
        old = monthly.loc[key, "Close"]
        if row.Role == "fallback" and pd.notna(old):
            raise ValueError(f"Fallback now overlaps Yahoo; review the join: {row.Evidence_ID}")
        if row.Role == "override" and pd.isna(old):
            raise ValueError(f"Monthly correction has no existing observation: {row.Evidence_ID}")
        monthly.loc[key, "Nominal_Close"] = row.Nominal_Close
        monthly.loc[key, "Evidence_ID"] = row.Evidence_ID
        if row.Role == "event_price":
            if pd.notna(row.Published_Date):
                monthly.loc[key, "Available_Date"] = max(row.Observation_Date, row.Published_Date)
            continue
        adjustment_end = row.Adjustment_Through
        if pd.isna(adjustment_end) or adjustment_end < row.Observation_Date:
            raise ValueError(f"Invalid monthly adjustment horizon: {row.Evidence_ID}")
        adjustment_rows = dividends.loc[dividends.Asset_ID.eq(row.Asset_ID)
                                         & dividends.Ex_Date.gt(row.Observation_Date)
                                         & dividends.Ex_Date.le(adjustment_end)]
        factor = float(adjustment_rows.Factor.prod())
        available = max(row.Observation_Date,
                        row.Published_Date if pd.notna(row.Published_Date) else row.Observation_Date)
        if not adjustment_rows.empty:
            available = max(available, adjustment_rows.Available_Date.max())
        if row.Adjustment_Method == "yahoo_anchor":
            if pd.isna(row.Anchor_Date) or row.Anchor_Date != adjustment_end:
                raise ValueError(f"Invalid monthly anchor date: {row.Evidence_ID}")
            if not np.isfinite(row.Anchor_Nominal_Close) or row.Anchor_Nominal_Close <= 0:
                raise ValueError(f"Invalid monthly nominal anchor: {row.Evidence_ID}")
            factor *= _daily_close(market, row.Asset_ID, row.Anchor_Date) / row.Anchor_Nominal_Close
            available = max(available, row.Anchor_Date)
        elif row.Adjustment_Method != "dividend_chain":
            raise ValueError(f"Price repair lacks an adjustment method: {row.Evidence_ID}")
        monthly.loc[key, ["Observation_Date", "Close", "Price_Source", "Adjustment_Factor", "Available_Date"]] = [
            row.Observation_Date, row.Nominal_Close * factor,
            OVERRIDE_SOURCE if row.Role == "override" else RECONSTRUCTED_SOURCE, factor, available,
        ]
    return validate_monthly_history(monthly.reset_index().loc[:, MONTHLY_COLUMNS])


def validate_monthly_history(frame):
    result = frame.copy()
    if tuple(result.columns) != MONTHLY_COLUMNS:
        raise ValueError("Prepared monthly market schema is inconsistent")
    _dates(result, ("Month", "Observation_Date", "Available_Date"))
    _positive(result, ("Close", "Nominal_Close", "Adjustment_Factor"), optional=("Close", "Nominal_Close", "Adjustment_Factor"))
    result["Volume"] = pd.to_numeric(result.Volume.replace("", np.nan), errors="raise")
    if (result.duplicated(["Month", "Asset_ID"]).any() or result.Asset_ID.eq("").any()
            or result[["Month", "Observation_Date", "Available_Date"]].isna().any().any()
            or not result.Month.dt.is_month_end.all()
            or not (result.Observation_Date + pd.offsets.MonthEnd(0)).eq(result.Month).all()
            or not result.Available_Date.ge(result.Observation_Date).all()
            or not result.Price_Source.isin({"yahoo", "yahoo_supplement", RECONSTRUCTED_SOURCE, OVERRIDE_SOURCE}).all()
            or ((~np.isfinite(result.Volume) | result.Volume.lt(0)) & result.Volume.notna()).any()):
        raise ValueError("Invalid monthly history identity, date, source or volume")
    repaired = result.Price_Source.isin({RECONSTRUCTED_SOURCE, OVERRIDE_SOURCE})
    if (result.loc[repaired, "Evidence_ID"].eq("").any() or
            result.loc[repaired, ["Close", "Nominal_Close", "Adjustment_Factor"]].isna().any().any() or
            not np.allclose(result.loc[repaired, "Close"], result.loc[repaired, "Nominal_Close"]
                            * result.loc[repaired, "Adjustment_Factor"], rtol=1e-12, atol=1e-12)):
        raise ValueError("Reconstructed monthly close lacks consistent adjustment provenance")
    return result.sort_values(["Month", "Asset_ID"], kind="stable").reset_index(drop=True)


def causal_monthly_rows(monthly, *, cutoff):
    """Validate availability before exposing a prepared monthly prefix."""
    label = pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
    rows = monthly.loc[monthly.Month.le(label)].copy()
    future = rows.Close.notna() & (rows.Observation_Date.gt(cutoff) | rows.Available_Date.gt(cutoff))
    if future.any():
        sample = rows.loc[future, ["Asset_ID", "Month", "Available_Date"]].head().to_dict("records")
        raise ValueError(f"Monthly history uses future evidence at {cutoff}: {sample}")
    return rows


def monthly_matrices(monthly, *, cutoff):
    """Expose causal price/volume matrices and their monthly source calendar."""
    rows = causal_monthly_rows(monthly, cutoff=cutoff)
    label = pd.Timestamp(cutoff) + pd.offsets.MonthEnd(0)
    close = rows.pivot(index="Month", columns="Asset_ID", values="Close").sort_index()
    if not close.empty:
        close = close.reindex(pd.date_range(close.index.min(), label, freq="ME", name="Month"))
    volume = rows.pivot(index="Month", columns="Asset_ID", values="Volume").reindex_like(close)
    calendar = rows.groupby("Month").Observation_Date.max().to_dict()
    return close, volume, calendar
