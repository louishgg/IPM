"""Canonical mixed-source backtest price contracts and deterministic merge.

Reuters remains the first-priority source.  The only permitted fallbacks are
Yahoo ``Close`` observations captured with ``auto_adjust=False`` and the
hash-pinned frozen WIKI Prices extract declared below.  Every fallback row is
bound to one effective-dated canonical mapping before it can reach a prepared
artifact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)
from portfolio_core.provider_identity import (
    CONTRIBUTED_PRICE_RESOLUTION_METHODS,
    ProviderIdentity,
)
from portfolio_core.security_identity import SecurityIdentityBundle
from portfolio_core.security_identity import parse_exact_decimal


WIKI_PROVIDER = "wiki"
WIKI_PARENT_URL = (
    "https://media.githubusercontent.com/media/kmfranz/trading_pairs/"
    "aff4c4f3b677b0434bfedbc12b4137facaf7a0bb/WIKI_PRICES.csv"
)
WIKI_PARENT_COMMIT = "aff4c4f3b677b0434bfedbc12b4137facaf7a0bb"
WIKI_PARENT_BYTES = 235_562_224
WIKI_PARENT_SHA256 = (
    "dd5127aae478d270150904fcbad6e96a42e461e13c3d48a1587edb9b89cea43e"
)
WIKI_PARENT_ROWS = 2_166_605
WIKI_PARENT_FIRST_DATE = "2014-01-02"
WIKI_PARENT_LAST_DATE = "2016-12-19"
WIKI_COLUMNS = (
    "ticker",
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ex-dividend",
    "split_ratio",
    "adj_open",
    "adj_high",
    "adj_low",
    "adj_close",
    "adj_volume",
)
YAHOO_RAW_COLUMNS = (
    "Date",
    "Asset_ID",
    "Source_Ticker",
    "Provider_Symbol",
    "Mapping_ID",
    "Close",
    "Volume",
    "Dividends",
    "Stock_Splits",
    "Capital_Gains",
)
PRICE_OBSERVATION_COLUMNS = (
    "Observation_Date",
    "Month_End",
    "Provider",
    "Provider_Symbol",
    "Mapping_ID",
    "Asset_ID",
    "Price_Close",
    "Volume",
)
PRICE_SOURCE_PRIORITY = {"reuters": 0, "yahoo": 1, WIKI_PROVIDER: 2}
REUTERS_REMAP_METHOD = "reviewed_local_reuters_identity"


def validate_price_acquisition_manifest(
    paths,
) -> None:
    """Validate the exact unified raw-price catalog and selected payloads."""

    price_paths = paths.price_sources
    artifact_paths = (
        price_paths.yahoo_close_csv,
        price_paths.wiki_extract_csv,
        price_paths.acquisition_status_csv,
        price_paths.readiness_csv,
    )
    relative_paths = {
        path: path.relative_to(paths.project_root).as_posix()
        for path in artifact_paths
    }
    validate_exact_manifest_catalog(
        price_paths.artifact_manifest_csv,
        scope="backtest",
        dataset="prices",
        expected_origins={
            relative_path: ArtifactOrigin.DOWNLOADED
            for relative_path in relative_paths.values()
        },
        base_dir=paths.project_root,
    )


def load_verified_price_observations(
    paths,
    *,
    bundle: SecurityIdentityBundle,
) -> pd.DataFrame:
    """Load the one manifest-verified normalized price observation catalog."""

    price_paths = paths.price_sources
    if price_paths.artifact_manifest_csv.is_file():
        validate_price_acquisition_manifest(paths)
    return load_price_observation_catalog(paths, bundle=bundle)


def load_price_observation_catalog(
    paths,
    *,
    bundle: SecurityIdentityBundle,
    yahoo_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Load retained source files into the canonical observation catalog."""

    price_paths = paths.price_sources
    yahoo = (
        yahoo_raw
        if yahoo_raw is not None
        else pd.read_csv(
            price_paths.yahoo_close_csv,
            keep_default_na=False,
            float_precision="round_trip",
        )
        if price_paths.yahoo_close_csv.is_file()
        else None
    )
    wiki = (
        pd.read_csv(
            price_paths.wiki_extract_csv,
            keep_default_na=False,
            dtype=str,
        )
        if price_paths.wiki_extract_csv.is_file()
        else None
    )
    return build_price_observation_catalog(
        pd.read_csv(paths.prices_csv, low_memory=False),
        mappings=bundle.provider_mappings,
        yahoo_raw=yahoo,
        wiki_raw=wiki,
        bundle=bundle,
    )


def validate_yahoo_close_rows(
    frame: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
) -> pd.DataFrame:
    """Validate the sole Yahoo raw schema and effective-dated identities."""
    if tuple(frame.columns) != YAHOO_RAW_COLUMNS:
        raise ValueError(
            f"Yahoo backtest prices must have exactly {list(YAHOO_RAW_COLUMNS)}"
        )
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="coerce")
    if result["Date"].isna().any():
        raise ValueError("Yahoo backtest prices contain an invalid Date")
    for column in ("Asset_ID", "Source_Ticker", "Provider_Symbol"):
        values = result[column].astype("string").str.strip()
        if values.isna().any() or values.eq("").any():
            raise ValueError(f"Yahoo backtest prices contain an empty {column}")
        result[column] = values.astype(str)
    mapping_ids = result["Mapping_ID"].astype("string").fillna("").str.strip()
    result["Mapping_ID"] = mapping_ids.astype(str)
    if result.empty:
        return result.loc[:, list(YAHOO_RAW_COLUMNS)]
    if result.duplicated(["Provider_Symbol", "Date"]).any():
        raise ValueError("Yahoo backtest prices contain duplicate symbol/date rows")
    result["Close"] = pd.to_numeric(result["Close"], errors="coerce")
    if not np.isfinite(result["Close"].to_numpy(dtype=float)).all():
        raise ValueError("Yahoo backtest prices contain a non-finite Close")
    for column in ("Volume", "Dividends", "Stock_Splits", "Capital_Gains"):
        result[column] = pd.to_numeric(
            result[column].replace("", pd.NA), errors="coerce"
        )
        values = result[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Yahoo backtest prices contain a non-finite {column}")
    if (result["Close"] <= 0).any():
        raise ValueError("Yahoo Close must be positive")
    for column in ("Volume", "Dividends", "Stock_Splits", "Capital_Gains"):
        if result[column].dropna().lt(0).any():
            raise ValueError(f"Yahoo {column} cannot be negative")

    if result["Mapping_ID"].eq("").any():
        raise ValueError("Every Yahoo price row requires a Mapping_ID")
    duplicates = mappings.loc[
        mappings["Mapping_ID"].duplicated(keep=False), "Mapping_ID"
    ]
    if not duplicates.empty:
        raise ValueError(f"Duplicate Yahoo Mapping_ID: {duplicates.iloc[0]}")
    by_id = mappings.set_index("Mapping_ID")
    unknown = sorted(set(result["Mapping_ID"]) - set(by_id.index))
    if unknown:
        raise ValueError(f"Unknown Yahoo Mapping_ID: {unknown[0]}")

    ids = result["Mapping_ID"]
    starts = pd.to_datetime(ids.map(by_id["Effective_Start"]), errors="raise")
    ends = pd.to_datetime(
        ids.map(by_id["Effective_End"]).replace("", pd.NA), errors="raise"
    )
    valid = ids.map(by_id["Provider"]).eq("yahoo")
    for column in ("Asset_ID", "Source_Ticker", "Provider_Symbol"):
        valid &= result[column].eq(ids.map(by_id[column]))
    valid &= result["Date"].ge(starts) & (
        ends.isna() | result["Date"].lt(ends)
    )
    if not valid.all():
        row = result.loc[~valid].iloc[0]
        raise ValueError(
            "Yahoo row falls outside its canonical identity: "
            f"{row['Mapping_ID']} on {row['Date'].date()}"
        )

    methods = ids.map(by_id["Resolution_Method"])
    action_fields = ("Volume", "Dividends", "Stock_Splits", "Capital_Gains")
    monthly = methods.eq("reviewed_yahoo_close_fallback")
    if result.loc[monthly, list(action_fields)].isna().any().any():
        raise ValueError("Yahoo monthly-price row lacks full fields")
    execution = methods.eq("reviewed_effective_symbol")
    if (result.loc[execution, "Volume"].isna() | result.loc[
        execution, "Volume"
    ].le(0)).any():
        raise ValueError("Yahoo exact-execution row lacks positive Volume")

    bounds = result.groupby("Mapping_ID", sort=False)["Date"].agg(["min", "max"])
    expected_first = pd.to_datetime(
        pd.Series(bounds.index.map(by_id["Local_First_Date"]), index=bounds.index)
        .replace("", pd.NA),
        errors="raise",
    )
    expected_last = pd.to_datetime(
        pd.Series(bounds.index.map(by_id["Local_Last_Date"]), index=bounds.index)
        .replace("", pd.NA),
        errors="raise",
    )
    invalid_bounds = expected_first.notna() & (
        bounds["min"].ne(expected_first) | bounds["max"].ne(expected_last)
    )
    if invalid_bounds.any():
        raise ValueError(
            "Yahoo observation bounds do not match "
            f"{invalid_bounds.index[invalid_bounds][0]}"
        )
    return result.sort_values(["Mapping_ID", "Date"], kind="stable").reset_index(
        drop=True
    )


def yahoo_normalization_requirements(
    bundle: SecurityIdentityBundle,
) -> list[dict[str, str]]:
    """Resolve each Yahoo normalization claim to one canonical mapping."""

    claim_columns = [
        "Normalization_Source_ID",
        "Event_ID",
        "From_Ticker",
        "Normalization_Provider_Symbol",
        "Normalization_Treatment_Date",
        "Normalization_Denominator",
    ]
    claims = bundle.legs.loc[
        bundle.legs["Executable_Share_Ratio"].ne("")
        & bundle.legs["Normalization_Provider"].eq("yahoo"),
        claim_columns,
    ].drop_duplicates()
    conflicting = claims["Normalization_Source_ID"].duplicated(keep=False)
    if conflicting.any():
        source_id = claims.loc[conflicting, "Normalization_Source_ID"].iloc[0]
        raise ValueError(
            f"Normalization source {source_id} has conflicting event-leg claims"
        )
    mappings = bundle.provider_mappings.loc[
        bundle.provider_mappings["Scope"].eq("backtest")
        & bundle.provider_mappings["Provider"].eq("yahoo")
        & bundle.provider_mappings["Review_Status"].eq("approved")
    ]
    rows: list[dict[str, str]] = []
    for claim in claims.sort_values("Normalization_Source_ID").to_dict("records"):
        source_id = str(claim["Normalization_Source_ID"])
        treatment_date = str(claim["Normalization_Treatment_Date"])
        matches = mappings.loc[
            mappings["Source_Ticker"].eq(str(claim["From_Ticker"]))
            & mappings["Provider_Symbol"].eq(
                str(claim["Normalization_Provider_Symbol"])
            )
        ]
        matches = matches.loc[[
            ProviderIdentity.from_mapping(mapping).is_effective(treatment_date)
            for mapping in matches.to_dict("records")
        ]]
        if len(matches) != 1:
            raise ValueError(
                "Normalization must resolve exactly one canonical mapping for "
                f"{source_id}; found {list(matches['Mapping_ID'].astype(str))}"
            )
        rows.append({
            "Normalization_Source_ID": str(source_id),
            "Mapping_ID": str(matches.iloc[0]["Mapping_ID"]),
            "Treatment_Date": treatment_date,
            "Denominator": str(claim["Normalization_Denominator"]),
        })
    return rows


def validate_yahoo_event_treatments(
    frame: pd.DataFrame,
    *,
    bundle: SecurityIdentityBundle,
) -> pd.DataFrame:
    """Validate Yahoo rows once, including every normalization treatment."""
    result = validate_yahoo_close_rows(
        frame,
        mappings=bundle.provider_mappings,
    )
    for requirement in yahoo_normalization_requirements(bundle):
        source_id = requirement["Normalization_Source_ID"]
        treatment = result.loc[
            result["Mapping_ID"].eq(requirement["Mapping_ID"])
            & result["Date"].eq(pd.Timestamp(requirement["Treatment_Date"]))
        ]
        if len(treatment) != 1:
            raise ValueError(
                f"Normalization source {source_id} requires exactly one raw "
                "Yahoo treatment observation"
            )
        observed = treatment.iloc[0]["Stock_Splits"]
        if pd.isna(observed) or parse_exact_decimal(
            str(observed),
            field=f"{source_id} raw Stock_Splits",
        ) != parse_exact_decimal(
            requirement["Denominator"],
            field=f"{source_id} normalization denominator",
        ):
            raise ValueError(
                f"Normalization source {source_id} raw Yahoo factor is incorrect"
            )
    return result


def validate_wiki_extract_rows(
    frame: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
) -> pd.DataFrame:
    """Validate WIKI rows and return their canonical identities."""
    if tuple(frame.columns) != WIKI_COLUMNS:
        raise ValueError(f"WIKI extract must have exactly {list(WIKI_COLUMNS)}")
    result = frame.copy()
    result["ticker"] = result["ticker"].astype("string").str.strip()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result.empty or result["ticker"].isna().any() or result["date"].isna().any():
        raise ValueError("WIKI extract is empty or contains invalid identities/dates")
    wiki_mappings = wiki_mapping_rows(mappings)
    by_symbol = wiki_mappings.set_index("Provider_Symbol")
    authorized = tuple(sorted(by_symbol.index.astype(str)))
    symbols = set(result["ticker"].astype(str))
    if symbols != set(authorized):
        raise ValueError(
            f"WIKI extract symbols must be exactly {list(authorized)}; "
            f"found {sorted(symbols)}"
        )
    if result.duplicated(["ticker", "date"]).any():
        raise ValueError("WIKI extract contains duplicate ticker/date rows")
    for column in ("low", "high", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"WIKI extract contains a non-finite {column}")
    if (result["close"] <= 0).any():
        raise ValueError("WIKI close must be positive")
    if ((result["close"] < result["low"]) | (result["close"] > result["high"])).any():
        raise ValueError("WIKI close must lie within the reported daily low/high")

    symbols = result["ticker"].astype(str)
    starts = pd.to_datetime(symbols.map(by_symbol["Effective_Start"]), errors="raise")
    ends = pd.to_datetime(
        symbols.map(by_symbol["Effective_End"]).replace("", pd.NA), errors="raise"
    )
    effective = result["date"].ge(starts) & (
        ends.isna() | result["date"].lt(ends)
    )
    if not effective.all():
        row = result.loc[~effective].iloc[0]
        raise ValueError(
            "WIKI row does not resolve to exactly one effective identity: "
            f"{row['ticker']} on {row['date'].date()}"
        )
    result["Asset_ID"] = symbols.map(by_symbol["Asset_ID"])
    result["Mapping_ID"] = symbols.map(by_symbol["Mapping_ID"])
    bounds = result.groupby("ticker", sort=True)["date"].agg(["min", "max"])
    for mapping in wiki_mappings.to_dict("records"):
        symbol = str(mapping["Provider_Symbol"])
        expected = (
            pd.Timestamp(mapping["Local_First_Date"]),
            pd.Timestamp(mapping["Local_Last_Date"]),
        )
        actual = (bounds.loc[symbol, "min"], bounds.loc[symbol, "max"])
        if actual != expected:
            raise ValueError(
                f"WIKI extract bounds for {symbol} must be "
                f"{expected[0].date()}..{expected[1].date()}; found "
                f"{actual[0].date()}..{actual[1].date()}"
            )
    return result.sort_values(["ticker", "date"], kind="stable").reset_index(
        drop=True
    )


def wiki_mapping_rows(mappings: pd.DataFrame) -> pd.DataFrame:
    """Return the approved WIKI fallback identity intervals."""
    required = {
        "Mapping_ID",
        "Scope",
        "Provider",
        "Source_Ticker",
        "Provider_Symbol",
        "Asset_ID",
        "Effective_Start",
        "Effective_End",
        "Event_ID",
        "Resolution_Method",
        "Local_First_Date",
        "Local_Last_Date",
        "Review_Status",
    }
    missing = required - set(mappings.columns)
    if missing:
        raise ValueError(f"Provider mappings lack WIKI fields: {sorted(missing)}")
    rows = mappings.loc[
        mappings["Scope"].eq("backtest")
        & mappings["Provider"].eq(WIKI_PROVIDER)
        & mappings["Resolution_Method"].eq("reviewed_wiki_close_fallback")
        & mappings["Review_Status"].eq("approved")
    ].copy()
    if rows.empty:
        raise ValueError("No approved WIKI mappings serve backtest consumers")
    if rows["Mapping_ID"].duplicated().any() or rows[
        "Provider_Symbol"
    ].duplicated().any():
        raise ValueError("Approved WIKI mappings must have unique IDs and symbols")
    if rows[["Local_First_Date", "Local_Last_Date"]].eq("").any().any():
        raise ValueError("Approved WIKI mappings require local extraction bounds")
    return rows.sort_values("Mapping_ID", kind="stable").reset_index(drop=True)


def build_price_observation_catalog(
    reuters_raw: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
    yahoo_raw: pd.DataFrame | None = None,
    wiki_raw: pd.DataFrame | None = None,
    bundle: SecurityIdentityBundle | None = None,
) -> pd.DataFrame:
    """Return the one provider-neutral catalog used by backtest consumers."""

    observations = [_normalize_reuters_rows(reuters_raw)]
    if yahoo_raw is not None:
        yahoo = (
            validate_yahoo_event_treatments(yahoo_raw, bundle=bundle)
            if bundle is not None
            else validate_yahoo_close_rows(yahoo_raw, mappings=mappings)
        )
        observations.append(pd.DataFrame({
            "Observation_Date": yahoo["Date"],
            "Provider": "yahoo",
            "Provider_Symbol": yahoo["Provider_Symbol"],
            "Mapping_ID": yahoo["Mapping_ID"],
            "Asset_ID": yahoo["Asset_ID"],
            "Price_Close": yahoo["Close"],
            "Volume": yahoo["Volume"],
        }))
    if wiki_raw is not None:
        wiki = validate_wiki_extract_rows(
            wiki_raw,
            mappings=mappings,
        )
        observations.append(pd.DataFrame({
            "Observation_Date": wiki["date"],
            "Provider": WIKI_PROVIDER,
            "Provider_Symbol": wiki["ticker"],
            "Mapping_ID": wiki["Mapping_ID"],
            "Asset_ID": wiki["Asset_ID"],
            "Price_Close": wiki["close"],
            "Volume": pd.to_numeric(wiki["volume"], errors="coerce"),
        }))
    catalog = pd.concat(observations, ignore_index=True)
    catalog["Observation_Date"] = pd.to_datetime(
        catalog["Observation_Date"], errors="raise"
    )
    catalog["Month_End"] = catalog["Observation_Date"] + pd.offsets.MonthEnd(0)
    return catalog.loc[
        :, list(PRICE_OBSERVATION_COLUMNS)
    ]


def _normalize_reuters_rows(raw: pd.DataFrame) -> pd.DataFrame:
    required = ("Date", "RIC", "Price Close")
    if not set(required).issubset(raw.columns):
        raise ValueError(f"Reuters prices must contain {list(required)}")
    columns = [*required, *(["Volume"] if "Volume" in raw.columns else [])]
    frame = raw.loc[:, columns].copy()
    if "Volume" not in frame:
        frame["Volume"] = np.nan
    raw_prices = frame["Price Close"].copy()
    raw_volumes = frame["Volume"].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["RIC"] = frame["RIC"].astype("string").str.strip()
    frame["Price Close"] = pd.to_numeric(frame["Price Close"], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    if frame["Date"].isna().any() or frame["RIC"].isna().any() or frame["RIC"].eq("").any():
        raise ValueError("Reuters prices contain an invalid Date or RIC")
    finite_price = frame["Price Close"].dropna().to_numpy(dtype=float)
    finite_volume = frame["Volume"].dropna().to_numpy(dtype=float)
    if (
        (raw_prices.notna() & frame["Price Close"].isna()).any()
        or (raw_volumes.notna() & frame["Volume"].isna()).any()
        or not np.isfinite(finite_price).all()
        or not np.isfinite(finite_volume).all()
        or (frame["Price Close"].dropna() <= 0).any()
        or (frame["Volume"].dropna() < 0).any()
    ):
        raise ValueError("Reuters prices contain invalid price/volume values")

    # The supplied artifact contains a small number of exact-price duplicates
    # where one row alone carries volume.  They are unambiguous only under this
    # strict consolidation rule; conflicting duplicate winners fail closed.
    duplicate_mask = frame.duplicated(["Date", "RIC"], keep=False)
    unique = frame.loc[~duplicate_mask].copy()
    consolidated: list[dict[str, object]] = []
    for (_, _), group in frame.loc[duplicate_mask].groupby(
        ["Date", "RIC"], sort=False, dropna=False
    ):
        prices = group["Price Close"].dropna().unique()
        volumes = group["Volume"].dropna().unique()
        if len(prices) > 1 or len(volumes) > 1:
            raise ValueError(
                "Reuters prices contain conflicting duplicate Date/RIC winners"
            )
        row = group.iloc[-1].to_dict()
        row["Price Close"] = prices[0] if len(prices) else np.nan
        row["Volume"] = volumes[0] if len(volumes) else np.nan
        consolidated.append(row)
    if consolidated:
        frame = pd.concat(
            [unique, pd.DataFrame(consolidated)], ignore_index=True
        ).sort_values(["Date", "RIC"], kind="stable").reset_index(drop=True)
    else:
        frame = unique.reset_index(drop=True)

    return pd.DataFrame({
        "Observation_Date": frame["Date"],
        "Provider": "reuters",
        "Provider_Symbol": frame["RIC"].astype(str),
        "Mapping_ID": "",
        "Asset_ID": frame["RIC"].astype(str),
        "Price_Close": frame["Price Close"],
        "Volume": frame["Volume"],
    })


def build_mixed_monthly_prices(
    observations: pd.DataFrame,
    *,
    mappings: pd.DataFrame,
) -> pd.DataFrame:
    """Select the latest usable monthly close under canonical source priority."""

    combined = observations.loc[:, list(PRICE_OBSERVATION_COLUMNS)].copy()
    monthly_mapping_ids = set(
        mappings.loc[
            mappings["Scope"].eq("backtest")
            & mappings["Resolution_Method"].isin(CONTRIBUTED_PRICE_RESOLUTION_METHODS)
            & mappings["Review_Status"].eq("approved"),
            "Mapping_ID",
        ].astype(str)
    )
    combined = combined.loc[
        combined["Provider"].eq("reuters")
        | combined["Mapping_ID"].isin(monthly_mapping_ids)
    ].copy()
    combined["Date"] = combined["Month_End"]
    combined["Priority"] = combined["Provider"].map(PRICE_SOURCE_PRIORITY)
    if combined["Priority"].isna().any():
        raise ValueError("Mixed price candidates contain an unsupported source")

    latest_date = combined.groupby(
        ["Date", "Asset_ID", "Provider"], sort=False
    )["Observation_Date"].transform("max")
    latest = combined.loc[combined["Observation_Date"].eq(latest_date)].copy()
    # Resolve recency within each source before availability. A missing latest
    # Reuters close makes Reuters unavailable for that asset-month; it must not
    # outrank a valid fallback or revive an older, stale Reuters observation.
    latest = latest.loc[latest["Price_Close"].notna()].copy()
    duplicate_source = latest.duplicated(
        ["Date", "Asset_ID", "Provider"], keep=False
    )
    if duplicate_source.any():
        values = latest.loc[
            duplicate_source, ["Date", "Asset_ID", "Provider", "Provider_Symbol"]
        ].to_dict("records")
        raise ValueError(f"Ambiguous same-source monthly price winners: {values}")

    best_priority = latest.groupby(["Date", "Asset_ID"], sort=False)[
        "Priority"
    ].transform("min")
    winners = latest.loc[latest["Priority"].eq(best_priority)].copy()
    duplicate_winner = winners.duplicated(["Date", "Asset_ID"], keep=False)
    if duplicate_winner.any():
        values = winners.loc[
            duplicate_winner, ["Date", "Asset_ID", "Provider", "Provider_Symbol"]
        ].to_dict("records")
        raise ValueError(f"Ambiguous cross-source monthly price winners: {values}")

    winners = winners.sort_values(["Date", "Asset_ID"], kind="stable").reset_index(
        drop=True
    )
    return winners.loc[:, ["Date", "Asset_ID", "Price_Close", "Volume"]].copy()


__all__ = [
    "PRICE_OBSERVATION_COLUMNS",
    "PRICE_SOURCE_PRIORITY",
    "REUTERS_REMAP_METHOD",
    "WIKI_COLUMNS",
    "WIKI_PARENT_BYTES",
    "WIKI_PARENT_COMMIT",
    "WIKI_PARENT_FIRST_DATE",
    "WIKI_PARENT_LAST_DATE",
    "WIKI_PARENT_ROWS",
    "WIKI_PARENT_SHA256",
    "WIKI_PARENT_URL",
    "WIKI_PROVIDER",
    "YAHOO_RAW_COLUMNS",
    "build_price_observation_catalog",
    "build_mixed_monthly_prices",
    "load_price_observation_catalog",
    "load_verified_price_observations",
    "validate_price_acquisition_manifest",
    "validate_wiki_extract_rows",
    "validate_yahoo_event_treatments",
    "validate_yahoo_close_rows",
    "wiki_mapping_rows",
    "yahoo_normalization_requirements",
]
