"""Read-only loaders for deterministic prepared backtest CSVs."""

from __future__ import annotations

import dataclasses
import csv
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from portfolio_core.dates import month_end_index
from portfolio_core.sector_assignments import (
    load_sector_assignments as load_prepared_sector_assignments,
)
from portfolio_core.corporate_actions import (
    EVENT_COLUMNS as CORPORATE_EVENT_COLUMNS,
    LEG_COLUMNS as CORPORATE_EVENT_LEG_COLUMNS,
    SOURCE_COLUMNS as CORPORATE_EVENT_SOURCE_COLUMNS,
    apply_corporate_actions,
)
from portfolio_core.portfolio_lifecycle import (
    EVENT_DELIVERY_EXECUTION_COLUMNS,
    eligible_assets_for_interval,
)
from portfolio_core.price_basis import (
    PriceBasisSpec,
    price_basis_spec,
    validate_price_basis,
)
from portfolio_core.security_identity import load_security_identity_bundle

from .config import (
    BACKTEST_BASELINE_END_DATE,
    BACKTEST_WARMUP_MONTHS,
    DEFAULT_CONFIG,
    BacktestMarketConfig,
)
from .paths import BacktestPaths
from .preparation_artifacts import validate_preparation_manifest
from .price_sources import (
    PRICE_OBSERVATION_COLUMNS,
    load_verified_price_observations,
)


PREPARE_CORE_COMMAND = "python -m backtest.prepare core"
VALIDATE_BENCHMARK_COMMAND = (
    "python -m data_acquisition.acquire backtest benchmark"
)
PREPARED_PRICE_COLUMNS = ("Date", "Asset_ID", "Price_Close", "Volume")
PREPARED_ASSET_METADATA_COLUMNS = (
    "Asset_ID",
    "Source_Ticker",
    "Source_Ticker_History",
    "Price_RIC",
    "Yahoo_Ticker",
    "Has_Price",
    "Has_PiT",
    "Is_Tradable",
    "Identity_Resolution_Status",
)
PREPARED_BENCHMARK_COLUMNS = ("Date", "SP500TR_Close")


@dataclasses.dataclass
class BacktestDataset:
    data_close: pd.DataFrame
    data_volume: pd.DataFrame
    pit_matrix: pd.DataFrame
    sector_assignments: pd.DataFrame
    valid_trading_days: pd.DatetimeIndex
    rolling_dollar_vol: pd.DataFrame
    asset_to_ticker: dict[str, str]
    unavailable_members: list[str] = dataclasses.field(default_factory=list)
    security_events: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    security_event_legs: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    security_event_sources: pd.DataFrame = dataclasses.field(default_factory=pd.DataFrame)
    event_delivery_executions: pd.DataFrame = dataclasses.field(
        default_factory=lambda: pd.DataFrame(
            columns=EVENT_DELIVERY_EXECUTION_COLUMNS
        )
    )
    price_basis: PriceBasisSpec = dataclasses.field(
        default_factory=lambda: price_basis_spec("backtest")
    )
    historical_asset_to_ticker: dict[str, str] = dataclasses.field(
        default_factory=dict
    )

    sector_return_history: object | None = None

    def __post_init__(self) -> None:
        """Require explicit display metadata for the canonical Asset-ID union."""
        if self.price_basis != price_basis_spec("backtest"):
            raise ValueError(
                f"Unsupported backtest price basis: {self.price_basis!r}"
            )
        expected = set(self.data_close.columns.astype(str)) | set(
            self.pit_matrix.columns.astype(str)
        )
        actual = {str(asset_id) for asset_id in self.asset_to_ticker}
        if actual != expected:
            raise ValueError(
                "asset_to_ticker must cover the complete price/PiT Asset-ID "
                f"union; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        empty = sorted(
            str(asset_id)
            for asset_id, ticker in self.asset_to_ticker.items()
            if not str(ticker).strip()
        )
        if empty:
            raise ValueError(f"asset_to_ticker has empty display values: {empty}")
        if self.sector_assignments.empty:
            raise ValueError("sector_assignments cannot be empty")

    def audit_ticker(self, asset_id: str, execution_date: object) -> str:
        """Return the frozen baseline label for pre-extension audit rows."""
        asset = str(asset_id)
        if pd.Timestamp(execution_date) <= pd.Timestamp(BACKTEST_BASELINE_END_DATE):
            return self.historical_asset_to_ticker.get(
                asset, self.asset_to_ticker[asset]
            )
        return self.asset_to_ticker[asset]


def active_pit_asset_ids(
    backtest_data: BacktestDataset,
    date: pd.Timestamp,
) -> list[str]:
    """Return point-in-time Asset_IDs active at or before ``date``."""
    return sorted(_active_pit_asset_ids(backtest_data.pit_matrix, date))


def _active_pit_asset_ids(
    pit_matrix: pd.DataFrame,
    date: object,
) -> frozenset[str]:
    """Return the active Asset-ID set from a prepared point-in-time matrix."""

    date = pd.Timestamp(date)
    eligible_dates = pit_matrix.index[pit_matrix.index <= date]
    if eligible_dates.empty:
        return frozenset()

    active_mask = pit_matrix.loc[eligible_dates[-1]]
    return frozenset(active_mask[active_mask].index.astype(str))


def derive_event_delivery_executions(
    *,
    events: pd.DataFrame,
    legs: pd.DataFrame,
    mappings: pd.DataFrame,
    observations: pd.DataFrame,
    pit_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """Derive off-universe delivery exits directly from canonical evidence."""

    deliveries = legs.loc[
        legs["Review_Status"].eq("approved")
        & legs["To_Asset_ID"].ne("")
        & legs["To_Asset_ID"].ne(legs["From_Asset_ID"])
    ]
    if deliveries.empty:
        return pd.DataFrame(columns=EVENT_DELIVERY_EXECUTION_COLUMNS)
    approved_events = events.loc[
        events["Review_Status"].eq("approved")
    ].set_index("Event_ID")
    execution_mappings = mappings.loc[
        mappings["Scope"].eq("backtest")
        & mappings["Resolution_Method"].eq("reviewed_effective_symbol")
        & mappings["Review_Status"].eq("approved")
        & mappings["Event_ID"].ne("")
    ]
    raw = observations.loc[:, list(PRICE_OBSERVATION_COLUMNS)].copy()
    raw["Observation_Date"] = pd.to_datetime(
        raw["Observation_Date"], errors="raise"
    ).dt.normalize()

    rows: list[dict[str, object]] = []
    for (event_id, asset_id), delivery_group in deliveries.groupby(
        ["Event_ID", "To_Asset_ID"], sort=True
    ):
        event_id = str(event_id)
        asset_id = str(asset_id)
        matches = execution_mappings.loc[
            execution_mappings["Event_ID"].eq(event_id)
            & execution_mappings["Asset_ID"].eq(asset_id)
        ]
        if matches.empty:
            continue
        if len(matches) != 1 or event_id not in approved_events.index:
            raise RuntimeError(
                f"Event-delivery execution identity is ambiguous: {event_id}/{asset_id}"
            )
        mapping = matches.iloc[0]
        bound = max(
            pd.Timestamp(approved_events.loc[event_id, "Effective_Date"]),
            pd.Timestamp(mapping["Effective_Start"]),
        ).normalize()
        mapped = raw.loc[
            raw["Mapping_ID"].eq(mapping["Mapping_ID"])
            & raw["Observation_Date"].ge(bound)
            & raw["Price_Close"].gt(0)
        ]
        effective_end = str(mapping["Effective_End"])
        if effective_end:
            mapped = mapped.loc[
                mapped["Observation_Date"].lt(pd.Timestamp(effective_end))
            ]
        if mapped.empty:
            raise RuntimeError(
                "Event-delivered tradable asset has no positive causal "
                f"observation: {event_id}/{asset_id}"
            )
        first = mapped.sort_values("Observation_Date", kind="stable").iloc[0]
        first_date = pd.Timestamp(first["Observation_Date"])
        if asset_id in _active_pit_asset_ids(pit_matrix, first_date):
            continue
        rows.append({
            "Event_ID": event_id,
            "From_Asset_IDs": tuple(sorted(set(
                delivery_group["From_Asset_ID"]
            ))),
            "Asset_ID": asset_id,
            "Execution_Date": first_date,
            "Reference_Close": float(first["Price_Close"]),
            "Volume": float(first["Volume"]),
        })

    result = pd.DataFrame(rows, columns=EVENT_DELIVERY_EXECUTION_COLUMNS)
    return result.sort_values(
        ["Execution_Date", "Event_ID", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)


def _prepared_data_error(path: Path, detail: str | None = None) -> RuntimeError:
    message = f"Prepared data file is unavailable or invalid: {path}."
    if detail:
        message += f" {detail}"
    message += f" Run `{PREPARE_CORE_COMMAND}` first."
    return RuntimeError(message)


def _require_prepared_file(path: Path) -> None:
    if not path.is_file():
        raise _prepared_data_error(path, "The file does not exist")


def load_price_basis(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> PriceBasisSpec:
    path = paths.price_basis_csv
    _require_prepared_file(path)
    try:
        frame = pd.read_csv(path, keep_default_na=False)
        return validate_price_basis(frame, "backtest")
    except (OSError, ValueError) as exc:
        raise _prepared_data_error(path, str(exc)) from exc


def load_prepared_benchmark(path: Path) -> pd.Series:
    """Parse the canonical prepared monthly S&P 500 total-return series."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, float_precision="round_trip")
    if tuple(frame.columns) != PREPARED_BENCHMARK_COLUMNS:
        raise ValueError(
            "Prepared benchmark must have columns "
            f"{list(PREPARED_BENCHMARK_COLUMNS)}; found {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Prepared benchmark is empty")
    dates = pd.to_datetime(frame["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Prepared benchmark contains an invalid Date")
    dates = pd.DatetimeIndex(dates).tz_localize(None)
    if dates.has_duplicates:
        raise ValueError("Prepared benchmark contains duplicate dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Prepared benchmark dates must be sorted ascending")
    if not dates.equals(dates + pd.offsets.MonthEnd(0)):
        raise ValueError("Prepared benchmark dates must all be month-end dates")
    values = pd.to_numeric(frame["SP500TR_Close"], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if values.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(
            "Prepared benchmark SP500TR_Close values must be finite and nonmissing"
        )
    if (numeric <= 0.0).any():
        raise ValueError(
            "Prepared benchmark SP500TR_Close values must be positive"
        )
    return pd.Series(
        numeric,
        index=dates,
        name="SP500TR_Close",
        dtype="float64",
    )


def validate_raw_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the Yahoo-derived monthly benchmark artifact."""
    if tuple(frame.columns) != PREPARED_BENCHMARK_COLUMNS:
        raise ValueError(
            "Raw benchmark must have columns "
            f"{list(PREPARED_BENCHMARK_COLUMNS)}; found {list(frame.columns)}"
        )
    if frame.empty:
        raise ValueError("Raw benchmark is empty")

    result = frame.copy()
    dates = pd.to_datetime(result["Date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Raw benchmark contains an invalid Date")
    dates = pd.DatetimeIndex(dates).tz_localize(None) + pd.offsets.MonthEnd(0)
    if dates.has_duplicates:
        raise ValueError("Raw benchmark contains duplicate monthly dates")

    values = pd.to_numeric(result["SP500TR_Close"], errors="coerce")
    numeric = values.to_numpy(dtype=float)
    if values.isna().any() or not np.isfinite(numeric).all():
        raise ValueError(
            "Raw benchmark SP500TR_Close values must be finite and nonmissing"
        )
    if (numeric <= 0.0).any():
        raise ValueError("Raw benchmark SP500TR_Close values must be positive")

    result["Date"] = dates
    result["SP500TR_Close"] = numeric
    return result.sort_values("Date", kind="stable").reset_index(drop=True)


def _parse_dates(values: pd.Series, path: Path) -> pd.DatetimeIndex:
    dates = pd.to_datetime(values, errors="coerce")
    if dates.isna().any():
        raise _prepared_data_error(path, "Date contains missing or invalid values")
    try:
        dates = dates.dt.tz_localize(None)
    except TypeError:
        dates = dates.dt.tz_convert(None)
    return pd.DatetimeIndex(dates)


def _coerce_bool_series(series: pd.Series, path: Path, column: str) -> pd.Series:
    if isinstance(series.dtype, pd.BooleanDtype) or series.dtype == bool:
        if series.isna().any():
            raise _prepared_data_error(path, f"{column} contains a missing boolean")
        return series.astype(bool)

    normalized = series.astype("string").str.strip().str.lower()
    parsed = normalized.map({"true": True, "false": False, "1": True, "0": False})
    if parsed.isna().any():
        invalid = sorted(normalized[parsed.isna()].dropna().unique().tolist())
        raise _prepared_data_error(
            path,
            f"{column} contains invalid booleans: {invalid[:5]}",
        )
    return parsed.astype(bool)


def _validate_monthly_index(
    index: pd.DatetimeIndex,
    market_config: BacktestMarketConfig,
    path: Path,
) -> None:
    if index.has_duplicates:
        raise _prepared_data_error(path, "Date contains duplicates")
    if not index.is_monotonic_increasing:
        raise _prepared_data_error(path, "Dates are not sorted")
    expected = month_end_index(market_config.start_date, market_config.end_date)
    if not index.equals(expected):
        raise _prepared_data_error(
            path,
            "Dates do not match the configured monthly data window",
        )


def _monthly_prefix(path: Path, end_date: object) -> StringIO:
    """Read only the dated prefix of a sorted prepared monthly CSV.

    Stop at the first later date before parsing any later numerical observations.
    The preparation manifest still verifies the complete file's integrity.
    """
    result = StringIO()
    writer = csv.writer(result)
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        date_column = header.index("Date")
        writer.writerow(header)
        for row in reader:
            if pd.Timestamp(row[date_column]) > pd.Timestamp(end_date):
                break
            writer.writerow(row)
    result.seek(0)
    return result


def load_price_data(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    *,
    development_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the complete prepared Date x Asset_ID price rectangle."""
    path = paths.prices_monthly_csv
    _require_prepared_file(path)
    print(f"Loading prepared month-end prices and volumes from {path}...")

    prices = pd.read_csv(
        _monthly_prefix(path, market_config.end_date) if development_only else path,
        dtype={"Asset_ID": "string"},
        float_precision="round_trip",
    )
    if tuple(prices.columns) != PREPARED_PRICE_COLUMNS:
        raise _prepared_data_error(
            path,
            "Expected columns "
            f"{list(PREPARED_PRICE_COLUMNS)}, found {list(prices.columns)}",
        )
    if prices.empty:
        raise _prepared_data_error(path, "The file is empty")

    prices["Date"] = _parse_dates(prices["Date"], path)
    if prices["Asset_ID"].isna().any():
        raise _prepared_data_error(path, "Asset_ID contains a missing value")
    prices["Asset_ID"] = prices["Asset_ID"].str.strip()
    if prices["Asset_ID"].eq("").any():
        raise _prepared_data_error(path, "Asset_ID contains an empty value")
    if prices.duplicated(["Date", "Asset_ID"]).any():
        raise _prepared_data_error(path, "Date and Asset_ID are not unique")

    sorted_prices = prices.sort_values(
        ["Date", "Asset_ID"], kind="stable"
    ).reset_index(drop=True)
    if not prices.reset_index(drop=True).equals(sorted_prices):
        raise _prepared_data_error(path, "Rows are not sorted by Date and Asset_ID")

    asset_ids = sorted(prices["Asset_ID"].unique().tolist())
    dates = pd.DatetimeIndex(sorted(prices["Date"].unique()))
    _validate_monthly_index(dates, market_config, path)
    expected_rows = len(dates) * len(asset_ids)
    if len(prices) != expected_rows:
        raise _prepared_data_error(
            path,
            "The prepared price data is not a complete Date x Asset_ID rectangle",
        )

    expected_pairs = pd.MultiIndex.from_product(
        [dates, asset_ids], names=["Date", "Asset_ID"]
    )
    actual_pairs = pd.MultiIndex.from_frame(prices[["Date", "Asset_ID"]])
    if not actual_pairs.equals(expected_pairs):
        raise _prepared_data_error(
            path,
            "The prepared price namespace differs across dates",
        )

    for column in ("Price_Close", "Volume"):
        numeric = pd.to_numeric(prices[column], errors="coerce")
        invalid = prices[column].notna() & numeric.isna()
        if invalid.any():
            raise _prepared_data_error(path, f"{column} contains text values")
        finite = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise _prepared_data_error(path, f"{column} contains non-finite values")
        prices[column] = numeric
    if (prices["Price_Close"].dropna() <= 0.0).any():
        raise _prepared_data_error(path, "Price_Close contains non-positive values")
    if (prices["Volume"].dropna() < 0.0).any():
        raise _prepared_data_error(path, "Volume contains negative values")

    indexed = prices.set_index(["Date", "Asset_ID"])
    data_close = indexed["Price_Close"].unstack("Asset_ID")
    data_volume = indexed["Volume"].unstack("Asset_ID")
    data_close = data_close.reindex(index=dates, columns=asset_ids)
    data_volume = data_volume.reindex(index=dates, columns=asset_ids)
    data_close.columns.name = "Asset_ID"
    data_volume.columns.name = "Asset_ID"
    return data_close, data_volume


def load_pit_universe(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    *,
    development_only: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Load prepared wide point-in-time membership without rebuilding it."""
    path = paths.pit_membership_csv
    _require_prepared_file(path)
    print(f"Loading prepared point-in-time membership from {path}...")

    pit = pd.read_csv(
        _monthly_prefix(path, market_config.end_date) if development_only else path
    )
    if pit.empty or "Date" not in pit.columns:
        raise _prepared_data_error(path, "Expected a Date column and membership rows")
    asset_ids = [str(column) for column in pit.columns if column != "Date"]
    if not asset_ids:
        raise _prepared_data_error(path, "No Asset_ID membership columns were found")
    if len(asset_ids) != len(set(asset_ids)):
        raise _prepared_data_error(path, "Asset_ID columns are not unique")
    if asset_ids != sorted(asset_ids):
        raise _prepared_data_error(path, "Asset_ID columns are not sorted")

    dates = _parse_dates(pit.pop("Date"), path)
    _validate_monthly_index(dates, market_config, path)
    pit.columns = asset_ids
    for asset_id in asset_ids:
        pit[asset_id] = _coerce_bool_series(pit[asset_id], path, asset_id)
    pit.index = dates
    pit.index.name = "Date"
    pit.columns.name = "Asset_ID"
    if (pit.sum(axis=1) == 0).any():
        raise _prepared_data_error(path, "At least one month has zero constituents")
    return pit, asset_ids


def load_asset_metadata(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Load static prepared Asset_ID/display-ticker metadata."""
    path = paths.asset_metadata_csv
    _require_prepared_file(path)
    metadata = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            "Asset_ID": "string",
            "Source_Ticker": "string",
            "Source_Ticker_History": "string",
            "Price_RIC": "string",
            "Yahoo_Ticker": "string",
            "Identity_Resolution_Status": "string",
        },
    )
    if tuple(metadata.columns) != PREPARED_ASSET_METADATA_COLUMNS:
        raise _prepared_data_error(
            path,
            "Expected columns "
            f"{list(PREPARED_ASSET_METADATA_COLUMNS)}, "
            f"found {list(metadata.columns)}",
        )
    if metadata.empty:
        raise _prepared_data_error(path, "The file is empty")

    required_text = (
        "Asset_ID",
        "Source_Ticker",
        "Source_Ticker_History",
        "Yahoo_Ticker",
        "Identity_Resolution_Status",
    )
    for column in required_text:
        if metadata[column].isna().any():
            raise _prepared_data_error(path, f"{column} contains a missing value")
        metadata[column] = metadata[column].str.strip()
        if metadata[column].eq("").any():
            raise _prepared_data_error(path, f"{column} contains an empty value")
    if metadata["Asset_ID"].duplicated().any():
        raise _prepared_data_error(path, "Asset_ID is not unique")
    if metadata["Asset_ID"].tolist() != sorted(metadata["Asset_ID"].tolist()):
        raise _prepared_data_error(path, "Asset_ID rows are not sorted")

    for column in ("Has_Price", "Has_PiT", "Is_Tradable"):
        metadata[column] = _coerce_bool_series(metadata[column], path, column)
    metadata["Price_RIC"] = metadata["Price_RIC"].astype("string").fillna(
        ""
    ).str.strip()
    priced = metadata["Has_Price"]
    if (
        metadata.loc[priced, "Price_RIC"].eq("").any()
        or not metadata.loc[priced, "Price_RIC"].equals(
            metadata.loc[priced, "Asset_ID"]
        )
    ):
        raise _prepared_data_error(
            path,
            "Priced assets must use their exact full RIC as Asset_ID and Price_RIC",
        )
    if metadata.loc[~priced, "Price_RIC"].ne("").any():
        raise _prepared_data_error(
            path,
            "Unpriced assets cannot claim a Price_RIC",
        )
    expected_tradable = metadata["Has_Price"] & metadata["Has_PiT"]
    if not metadata["Is_Tradable"].equals(expected_tradable):
        raise _prepared_data_error(
            path,
            "Is_Tradable must equal Has_Price and Has_PiT",
        )
    return metadata


def load_sector_assignments(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Load exact-date sector rows and convert errors to preparation guidance."""
    try:
        return load_prepared_sector_assignments(paths.sector_assignments_csv)
    except (FileNotFoundError, ValueError) as exc:
        raise _prepared_data_error(paths.sector_assignments_csv, str(exc)) from exc


def load_security_event_data(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load prepared, scope-resolved event inputs without consulting raw URLs."""

    artifacts = (
        (paths.security_events_prepared_csv, CORPORATE_EVENT_COLUMNS, "events"),
        (
            paths.security_event_legs_prepared_csv,
            CORPORATE_EVENT_LEG_COLUMNS,
            "event legs",
        ),
        (
            paths.security_event_sources_prepared_csv,
            CORPORATE_EVENT_SOURCE_COLUMNS,
            "event sources",
        ),
    )
    loaded: list[pd.DataFrame] = []
    for path, columns, label in artifacts:
        _require_prepared_file(path)
        frame = pd.read_csv(path, keep_default_na=False)
        if list(frame.columns) != list(columns):
            raise _prepared_data_error(
                path,
                f"Prepared security {label} must have columns {list(columns)}",
            )
        loaded.append(frame)

    events, legs, sources = loaded
    if not events.empty:
        events["Effective_Date"] = _parse_dates(
            events["Effective_Date"], paths.security_events_prepared_csv
        )
    audit_path = paths.security_event_crossing_audit_csv
    _require_prepared_file(audit_path)
    audit = pd.read_csv(audit_path, keep_default_na=False)
    required_audit = {
        "Event_ID",
        "Effective_Date",
        "From_Asset_ID",
        "Potential_Holding_Crossing",
        "Accounting_Treatment",
        "Validation_Status",
    }
    if not required_audit.issubset(audit.columns):
        raise _prepared_data_error(
            audit_path,
            "Security-event crossing audit is missing required columns",
        )
    if audit.duplicated(["Event_ID", "From_Asset_ID"]).any():
        raise _prepared_data_error(
            audit_path,
            "Event_ID and From_Asset_ID rows are not unique",
        )
    for column in ("Potential_Holding_Crossing",):
        audit[column] = _coerce_bool_series(audit[column], audit_path, column)
    unresolved = audit.loc[
        audit["Potential_Holding_Crossing"]
        & ~audit["Validation_Status"].eq("approved"),
        "Event_ID",
    ].astype(str).tolist()
    if unresolved:
        raise _prepared_data_error(
            audit_path,
            f"Holding-relevant security events remain unresolved: {unresolved}",
        )
    try:
        apply_corporate_actions(
            {},
            {},
            events,
            legs,
            sources,
            start_exclusive="1900-01-01",
            end_inclusive="2100-12-31",
        )
    except ValueError as error:
        raise _prepared_data_error(
            paths.security_events_prepared_csv,
            f"Prepared corporate-action tables are inconsistent: {error}",
        ) from error
    return events, legs, sources


def assemble_backtest_dataset(
    *,
    data_close: pd.DataFrame,
    data_volume: pd.DataFrame,
    pit_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    sector_assignments: pd.DataFrame,
    security_events: pd.DataFrame,
    security_event_legs: pd.DataFrame,
    security_event_sources: pd.DataFrame,
    event_delivery_executions: pd.DataFrame,
    price_basis: PriceBasisSpec,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> BacktestDataset:
    """Validate loaded frames and assemble the canonical numerical dataset."""
    price_asset_ids = set(data_close.columns.astype(str))
    pit_asset_ids = set(pit_matrix.columns.astype(str))
    metadata_asset_ids = set(metadata["Asset_ID"])
    if metadata_asset_ids != price_asset_ids | pit_asset_ids:
        raise _prepared_data_error(
            paths.asset_metadata_csv,
            "Asset_ID coverage does not equal the price/PiT union",
        )

    metadata_by_asset = metadata.set_index("Asset_ID")
    has_price = set(metadata.loc[metadata["Has_Price"], "Asset_ID"])
    has_pit = set(metadata.loc[metadata["Has_PiT"], "Asset_ID"])
    if has_price != price_asset_ids or has_pit != pit_asset_ids:
        raise _prepared_data_error(
            paths.asset_metadata_csv,
            "Has_Price or Has_PiT disagrees with prepared CSV namespaces",
        )

    tradable_asset_ids = metadata.loc[
        metadata["Is_Tradable"], "Asset_ID"
    ].tolist()
    expected_tradable = sorted(price_asset_ids & pit_asset_ids)
    if tradable_asset_ids != expected_tradable:
        raise _prepared_data_error(
            paths.asset_metadata_csv,
            "Tradable Asset_IDs do not equal the price/PiT intersection",
        )
    print(
        "Intersection complete: Reduced universe to "
        f"{len(tradable_asset_ids)} valid tradable assets."
    )

    asset_to_ticker = metadata_by_asset["Source_Ticker"].astype(str).to_dict()
    resolution = pd.read_csv(
        paths.ticker_ric_resolution_csv,
        usecols=["Date", "Asset_ID", "Source_Ticker"],
        keep_default_na=False,
    )
    resolution["Date"] = pd.to_datetime(resolution["Date"], errors="raise")
    baseline_resolution = resolution.loc[
        resolution["Date"].le(pd.Timestamp(BACKTEST_BASELINE_END_DATE))
        & resolution["Asset_ID"].astype(str).ne("")
    ].sort_values(["Date", "Source_Ticker"], kind="stable")
    historical_asset_to_ticker = (
        baseline_resolution.drop_duplicates("Asset_ID", keep="last")
        .set_index("Asset_ID")["Source_Ticker"]
        .astype(str)
        .to_dict()
    )
    expected_sector_dates = data_close.index[BACKTEST_WARMUP_MONTHS:-1]
    expected_assignment_dates = expected_sector_dates
    actual_assignment_dates = pd.DatetimeIndex(
        sector_assignments["As_Of_Date"].drop_duplicates()
    ).sort_values()
    if not actual_assignment_dates.equals(expected_assignment_dates):
        raise _prepared_data_error(
            paths.sector_assignments_csv,
            "As_Of_Date rows do not match the monthly sector consumers",
        )
    assigned_asset_ids = set(sector_assignments["Asset_ID"].astype(str))
    if not assigned_asset_ids.issubset(metadata_asset_ids):
        raise _prepared_data_error(
            paths.sector_assignments_csv,
            "Sector assignments contain Asset_IDs outside prepared metadata",
        )
    lifecycle = SimpleNamespace(
        data_close=data_close,
        security_events=security_events,
        security_event_legs=security_event_legs,
        security_event_sources=security_event_sources,
        event_delivery_executions=event_delivery_executions,
    )
    for date, next_date in zip(
        expected_sector_dates,
        data_close.index[BACKTEST_WARMUP_MONTHS + 1:],
    ):
        eligible_dates = pit_matrix.index[pit_matrix.index <= date]
        active_mask = pit_matrix.loc[eligible_dates[-1]]
        consumers = set(
            eligible_assets_for_interval(
                lifecycle,
                active_mask[active_mask].index,
                pd.Timestamp(date),
                pd.Timestamp(next_date),
            )
        )
        assigned = set(
            sector_assignments.loc[
                sector_assignments["As_Of_Date"].eq(date), "Asset_ID"
            ].astype(str)
        )
        if consumers != assigned:
            missing = sorted(consumers - assigned)
            unexpected = sorted(assigned - consumers)
            raise _prepared_data_error(
                paths.sector_assignments_csv,
                "Sector assignments do not exactly cover canonical consumers "
                f"on {date.date()}; missing={missing[:20]}, "
                f"unexpected={unexpected[:20]}",
            )
    unavailable_members = metadata.loc[
        metadata["Has_PiT"] & ~metadata["Has_Price"], "Asset_ID"
    ].tolist()
    valid_trading_days = data_close.index[BACKTEST_WARMUP_MONTHS:]
    rolling_dollar_vol = (data_close * data_volume).rolling(
        window=3,
        min_periods=1,
    ).median()

    return BacktestDataset(
        data_close=data_close,
        data_volume=data_volume,
        pit_matrix=pit_matrix,
        sector_assignments=sector_assignments,
        valid_trading_days=valid_trading_days,
        rolling_dollar_vol=rolling_dollar_vol,
        asset_to_ticker=asset_to_ticker,
        unavailable_members=unavailable_members,
        security_events=security_events,
        security_event_legs=security_event_legs,
        security_event_sources=security_event_sources,
        event_delivery_executions=event_delivery_executions,
        price_basis=price_basis,
        historical_asset_to_ticker=historical_asset_to_ticker,
    )


def load_backtest_data(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    *,
    include_brinson: bool = False,
    check_raw: bool = False,
    development_only: bool = False,
) -> BacktestDataset:
    """Load prepared inputs after validating the requested manifest scope.

    development_only caps valuation inputs at the configured validation end and
    rejects include_brinson. Integrity checks may still read whole source files;
    the cutoff limits data used for evaluation, not bytes read for verification.
    """
    if development_only and include_brinson:
        raise ValueError("Development loading cannot include test-window Brinson inputs")
    cutoff = pd.Timestamp(DEFAULT_CONFIG.evaluation.validation_end_date)
    if development_only:
        market_config = dataclasses.replace(market_config, end_date=cutoff.date())
    yahoo_path = paths.price_sources.yahoo_close_csv
    directly_consumed_raw = (
        {
            "fallback_price_artifact_manifest",
            "security_identity_manifest",
        }
        if yahoo_path.is_file()
        else set()
    )
    validate_preparation_manifest(
        paths=paths,
        include_brinson=include_brinson,
        check_raw=check_raw,
        validate_raw_artifacts=directly_consumed_raw,
    )
    scope = {"development_only": True} if development_only else {}
    data_close, data_volume = load_price_data(market_config, paths, **scope)
    pit_matrix, _ = load_pit_universe(market_config, paths, **scope)
    metadata = load_asset_metadata(paths)
    sector_assignments = load_sector_assignments(paths)
    security_events, security_event_legs, security_event_sources = (
        load_security_event_data(paths)
    )
    if development_only:
        # The final endpoint is for valuation, never a construction snapshot.
        sector_assignments = sector_assignments.loc[
            sector_assignments.As_Of_Date.lt(cutoff)
        ].copy()
        security_events = security_events.loc[
            security_events.Effective_Date.le(cutoff)
        ].copy()
        event_ids = set(security_events.Event_ID)
        security_event_legs = security_event_legs.loc[
            security_event_legs.Event_ID.isin(event_ids)
        ].copy()
        security_event_sources = security_event_sources.loc[
            security_event_sources.Event_ID.isin(event_ids)
        ].copy()
    identity = load_security_identity_bundle(paths.project_root)
    if yahoo_path.is_file():
        price_observations = load_verified_price_observations(
            paths,
            bundle=identity,
        )
    else:
        yahoo_mappings = identity.provider_mappings.loc[
            identity.provider_mappings["Scope"].astype(str).eq("backtest")
            & identity.provider_mappings["Provider"].astype(str).eq("yahoo")
            & identity.provider_mappings["Review_Status"].astype(str).eq(
                "approved"
            )
        ]
        if not yahoo_mappings.empty:
            raise _prepared_data_error(
                yahoo_path,
                "Approved backtest Yahoo mappings require the unified raw table",
            )
        price_observations = pd.DataFrame(columns=PRICE_OBSERVATION_COLUMNS)
    price_basis = load_price_basis(paths)
    if development_only:
        # Provider files are verified as source artifacts; later observations
        # must never enter delivery selection or interval valuation.
        price_observations = price_observations.loc[
            pd.to_datetime(price_observations.Observation_Date).le(cutoff)
        ].copy()
    event_delivery_executions = derive_event_delivery_executions(
        events=security_events,
        legs=security_event_legs,
        mappings=identity.provider_mappings,
        observations=price_observations,
        pit_matrix=pit_matrix,
    )

    dataset = assemble_backtest_dataset(
        data_close=data_close,
        data_volume=data_volume,
        pit_matrix=pit_matrix,
        metadata=metadata,
        sector_assignments=sector_assignments,
        security_events=security_events,
        security_event_legs=security_event_legs,
        security_event_sources=security_event_sources,
        event_delivery_executions=event_delivery_executions,
        price_basis=price_basis,
        paths=paths,
    )
    return dataset
