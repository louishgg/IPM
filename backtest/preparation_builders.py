"""Offline preparation of deterministic CSV inputs for backtest analysis."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from data_acquisition.contracts import (
    ProviderStatus,
    ReadinessStatus,
    read_acquisition_statuses,
    read_readiness,
)
from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)
from portfolio_core.io import atomic_write_dataframe
from portfolio_core.price_basis import price_basis_frame
from portfolio_core.dates import month_end_index
from portfolio_core.provider_identity import (
    CONTRIBUTED_PRICE_RESOLUTION_METHODS,
    YahooIdentityResolver,
)
from portfolio_core.sector_resolution import (
    build_sector_assignments_from_evidence,
)
from portfolio_core.security_identity import (
    load_security_identity_bundle,
)

from .config import (
    DEFAULT_CONFIG,
    BacktestBenchmarkConfig,
    BacktestMarketConfig,
)
from .data_loading import (
    BacktestDataset,
    PREPARED_ASSET_METADATA_COLUMNS,
    PREPARED_PRICE_COLUMNS,
    VALIDATE_BENCHMARK_COMMAND,
    load_backtest_data,
    load_prepared_benchmark,
    load_price_data,
    load_security_event_data,
    validate_raw_benchmark,
)
from .membership_resolution import build_membership_resolution_outputs
from .paths import BacktestPaths
from .preparation_artifacts import PREPARE_ALL_COMMAND
from .security_event_preparation import prepare_backtest_security_events
from .price_sources import (
    build_mixed_monthly_prices,
    load_verified_price_observations,
)
from .sector_requirements import (
    consumer_sector_requirements_from_resolution,
)


def _display_ticker_from_price_ric(value) -> str:
    """Return a display ticker, never an internal asset identity."""
    return str(value).strip().split(".", 1)[0].split("^", 1)[0]


def _normalize_asset_ids(values: Iterable) -> tuple[str, ...]:
    normalized = set()
    for value in values:
        if pd.isna(value):
            continue
        asset_id = str(value).strip()
        if asset_id:
            normalized.add(asset_id)
    return tuple(sorted(normalized))


def _require_raw_file(path: Path, *, recovery: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing raw data file: {path}. {recovery}"
        )


def prepare_prices_monthly(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Prepare the one canonical Reuters/Yahoo/WIKI monthly price rectangle."""
    _require_raw_file(
        paths.prices_csv,
        recovery=(
            "Restore the supplied Reuters price artifact under "
            "`data/shared/supplied/reuters/`; `python -m data_acquisition.acquire backtest "
            "prices` validates that local file but does not reacquire it."
        ),
    )
    identity_bundle = load_security_identity_bundle(
        paths.project_root,
        validate_manifest=True,
        require_all_approved=True,
    )
    mapping_catalog = identity_bundle.provider_mappings
    fallback_required = bool(
        (
            mapping_catalog["Scope"].eq("backtest")
            & mapping_catalog["Resolution_Method"].isin(
                CONTRIBUTED_PRICE_RESOLUTION_METHODS
            )
        ).any()
    )
    price_paths = paths.price_sources
    if fallback_required:
        required_raw = (
            price_paths.yahoo_close_csv,
            price_paths.wiki_extract_csv,
            price_paths.acquisition_status_csv,
            price_paths.readiness_csv,
            price_paths.artifact_manifest_csv,
        )
        missing_raw = [str(path) for path in required_raw if not path.is_file()]
        if missing_raw:
            raise FileNotFoundError(
                "Missing canonical fallback-price artifacts; run `python -m "
                "data_acquisition.acquire backtest prices`: "
                f"{missing_raw}"
            )
    readiness = (
        read_readiness(price_paths.readiness_csv)
        if price_paths.readiness_csv.is_file()
        else []
    )
    observations = load_verified_price_observations(
        paths,
        bundle=identity_bundle,
    )
    sparse = build_mixed_monthly_prices(
        observations,
        mappings=mapping_catalog,
    )
    if fallback_required:
        from .acquisition_planning import (
            price_requirement_ledger,
            validate_price_readiness_records,
        )

        ledger = price_requirement_ledger(
            paths,
            market_config=market_config,
            bundle=identity_bundle,
        )
        validate_price_readiness_records(
            ledger,
            observations,
            readiness,
            mappings=mapping_catalog,
        )
        if any(item.status is not ReadinessStatus.COMPLETE for item in readiness):
            raise ValueError(
                "Fallback-price causal readiness is incomplete; rerun the price "
                "acquisition command"
            )
    dates = month_end_index(market_config.start_date, market_config.end_date)
    sparse = sparse.loc[sparse["Date"].isin(dates)].copy()
    asset_ids = list(_normalize_asset_ids(sparse["Asset_ID"]))
    grouped = (
        sparse.sort_values(["Date", "Asset_ID"], kind="stable")
        .set_index(["Date", "Asset_ID"])[["Price_Close", "Volume"]]
    )
    complete_index = pd.MultiIndex.from_product(
        [dates, asset_ids], names=["Date", "Asset_ID"]
    )
    prepared = grouped.reindex(complete_index).reset_index()
    prepared = prepared.loc[:, list(PREPARED_PRICE_COLUMNS)]
    atomic_write_dataframe(
        prepared,
        paths.prices_monthly_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return prepared


def build_sector_assignment_requirements(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Return the strategy-independent exact-date sector consumer graph."""
    if not paths.ticker_ric_resolution_csv.is_file():
        raise FileNotFoundError(
            f"Missing prepared membership resolution: {paths.ticker_ric_resolution_csv}"
        )
    resolution = pd.read_csv(
        paths.ticker_ric_resolution_csv,
        keep_default_na=False,
    )
    resolution["Date"] = pd.to_datetime(resolution["Date"], errors="raise")
    data_close, _ = load_price_data(market_config, paths)
    events, event_legs, event_sources = load_security_event_data(paths)
    return consumer_sector_requirements_from_resolution(
        resolution,
        data_close,
        security_events=events,
        security_event_legs=event_legs,
        security_event_sources=event_sources,
        market_config=market_config,
    )


def prepare_pit_membership(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> tuple[pd.DataFrame, list[str]]:
    pit, asset_ids, resolution, coverage_audit, evidence_validation = (
        build_membership_resolution_outputs(market_config, paths)
    )
    atomic_write_dataframe(
        pit.reset_index(),
        paths.pit_membership_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    atomic_write_dataframe(
        resolution,
        paths.ticker_ric_resolution_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    atomic_write_dataframe(
        coverage_audit,
        paths.membership_coverage_audit_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    atomic_write_dataframe(
        evidence_validation,
        paths.security_identity_validation_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return pit, asset_ids


def prepare_asset_metadata(
    price_asset_ids: Iterable[str],
    pit_asset_ids: Iterable[str],
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Prepare static identity metadata without time-varying classifications."""
    price_ids = set(_normalize_asset_ids(price_asset_ids))
    pit_ids = set(_normalize_asset_ids(pit_asset_ids))
    yahoo_resolver = YahooIdentityResolver(
        load_security_identity_bundle(
            paths.project_root,
            validate_manifest=True,
            require_all_approved=True,
        ),
        scope="backtest",
    )
    if not paths.ticker_ric_resolution_csv.is_file():
        raise FileNotFoundError(
            "Missing prepared membership ticker/RIC resolution: "
            f"{paths.ticker_ric_resolution_csv}"
        )
    resolution = pd.read_csv(
        paths.ticker_ric_resolution_csv,
        keep_default_na=False,
        dtype={
            "Source_Ticker": "string",
            "Asset_ID": "string",
            "Price_RIC": "string",
            "Resolution_Method": "string",
        },
    )
    resolution["Date"] = pd.to_datetime(resolution["Date"], errors="raise")
    by_asset = {
        str(asset_id): group.sort_values(
            ["Date", "Source_Ticker"], kind="stable"
        )
        for asset_id, group in resolution.groupby("Asset_ID", sort=True)
    }
    rows = []
    for asset_id in sorted(price_ids | pit_ids):
        resolved = by_asset.get(asset_id)
        if resolved is None:
            source_tickers = [
                _display_ticker_from_price_ric(str(asset_id).strip())
            ]
            source_ticker = source_tickers[-1]
            price_ric = asset_id if asset_id in price_ids else ""
            resolution_status = "price_only_not_in_fja"
        else:
            source_tickers = sorted(
                resolved["Source_Ticker"].drop_duplicates().astype(str)
            )
            source_ticker = str(resolved.iloc[-1]["Source_Ticker"])
            price_rics = sorted(
                value
                for value in resolved["Price_RIC"].drop_duplicates().astype(str)
                if value
            )
            if len(price_rics) > 1 or (
                price_rics and price_rics[0] != asset_id
            ):
                raise ValueError(
                    f"Asset_ID {asset_id} has inconsistent Price_RIC history: "
                    f"{price_rics}"
                )
            price_ric = price_rics[0] if price_rics else ""
            resolution_status = ";".join(sorted(
                resolved["Resolution_Method"].drop_duplicates().astype(str)
            ))

        has_price = asset_id in price_ids
        has_pit = asset_id in pit_ids
        rows.append({
            "Asset_ID": asset_id,
            "Source_Ticker": source_ticker,
            "Source_Ticker_History": ";".join(source_tickers),
            "Price_RIC": price_ric,
            "Yahoo_Ticker": yahoo_resolver.resolve(
                source_ticker,
                purpose="historical_prices",
            ).provider_symbol,
            "Has_Price": has_price,
            "Has_PiT": has_pit,
            "Is_Tradable": has_price and has_pit,
            "Identity_Resolution_Status": resolution_status,
        })
    metadata = pd.DataFrame(rows, columns=PREPARED_ASSET_METADATA_COLUMNS)
    atomic_write_dataframe(
        metadata,
        paths.asset_metadata_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return metadata


def prepare_sector_assignments(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Resolve every active or generated execution need to a dated GICS row."""
    requirements = build_sector_assignment_requirements(market_config, paths)
    assignments = build_sector_assignments_from_evidence(
        requirements,
        paths=paths.sector_history,
        repository_root=paths.project_root,
    )
    atomic_write_dataframe(
        assignments,
        paths.sector_assignments_csv,
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )
    return assignments


def prepare_benchmark_monthly(
    benchmark_config: BacktestBenchmarkConfig = DEFAULT_CONFIG.benchmark,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
) -> pd.DataFrame:
    """Validate the acquired S&P 500 total-return month-end data."""
    _require_raw_file(
        paths.benchmark_raw_csv,
        recovery=(
            "Acquire the canonical Yahoo benchmark with "
            f"`{VALIDATE_BENCHMARK_COMMAND}`."
        ),
    )
    benchmark = validate_raw_benchmark(pd.read_csv(paths.benchmark_raw_csv))
    required_dates = benchmark_config.required_dates
    available_dates = pd.DatetimeIndex(benchmark["Date"])
    missing_dates = required_dates.difference(available_dates)
    if len(missing_dates):
        raise ValueError(
            "Benchmark input does not cover the configured data window; "
            f"missing {missing_dates.strftime('%Y-%m-%d').tolist()}"
        )
    validate_exact_manifest_catalog(
        paths.benchmark_artifact_manifest_csv,
        scope="backtest",
        dataset="benchmark",
        expected_origins={
            paths.benchmark_raw_csv.relative_to(paths.project_root).as_posix(): (
                ArtifactOrigin.DOWNLOADED
            ),
            paths.benchmark_acquisition_status_csv.relative_to(
                paths.project_root
            ).as_posix(): ArtifactOrigin.DOWNLOADED,
            paths.benchmark_readiness_csv.relative_to(
                paths.project_root
            ).as_posix(): ArtifactOrigin.DOWNLOADED,
        },
        base_dir=paths.project_root,
    )
    statuses = read_acquisition_statuses(paths.benchmark_acquisition_status_csv)
    if len(statuses) != 1:
        raise ValueError("Benchmark acquisition status must contain exactly one row")
    status = statuses[0]
    if not (
        status.identity.scope == "backtest"
        and status.identity.dataset == "benchmark"
        and status.identity.asset_id == "SP500TR"
        and status.identity.provider == "yahoo"
        and status.identity.provider_symbol == "^SP500TR"
        and status.status is ProviderStatus.OK
        and status.requested_start == benchmark_config.coverage_start_date.isoformat()
        and status.requested_end == benchmark_config.coverage_end_date.isoformat()
    ):
        raise ValueError(
            "Benchmark provider status is not a successful active-window "
            "Yahoo ^SP500TR acquisition"
        )
    readiness = read_readiness(paths.benchmark_readiness_csv)
    if len(readiness) != 1:
        raise ValueError("Benchmark readiness must contain exactly one row")
    record = readiness[0]
    if not (
        record.scope == "backtest"
        and record.dataset == "benchmark"
        and record.asset_id == "SP500TR"
        and record.requirement_set == "monthly_total_return_benchmark"
        and record.required_count == len(required_dates)
        and record.covered_count == len(required_dates)
        and record.status is ReadinessStatus.COMPLETE
        and record.contributing_sources == ("yahoo",)
    ):
        raise ValueError(
            "Benchmark readiness is not complete for the active Yahoo window"
        )
    atomic_write_dataframe(
        benchmark,
        paths.benchmark_csv,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.17g",
        lineterminator="\n",
    )
    return benchmark


def _validate_prepared_benchmark(
    path: Path,
    benchmark_config: BacktestBenchmarkConfig,
) -> None:
    try:
        benchmark = load_prepared_benchmark(path)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"Invalid prepared benchmark at {path}. "
            f"Run `{PREPARE_ALL_COMMAND}` first."
        ) from exc
    missing_dates = benchmark_config.required_dates.difference(benchmark.index)
    if len(missing_dates):
        raise RuntimeError(
            f"Prepared benchmark does not cover the configured data window at "
            f"{path}. Run `{PREPARE_ALL_COMMAND}` first."
        )


def validate_prepared_data(
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    benchmark_config: BacktestBenchmarkConfig = DEFAULT_CONFIG.benchmark,
    *,
    include_brinson: bool = False,
    check_raw: bool = False,
) -> BacktestDataset:
    """Return prepared data after validating the requested manifest scope."""
    backtest_data = load_backtest_data(
        market_config,
        paths,
        include_brinson=include_brinson,
        check_raw=check_raw,
    )
    _validate_prepared_benchmark(paths.benchmark_csv, benchmark_config)
    return backtest_data


def prepare_core_data(
    market_config: BacktestMarketConfig = DEFAULT_CONFIG.market,
    paths: BacktestPaths = DEFAULT_CONFIG.paths,
    benchmark_config: BacktestBenchmarkConfig = DEFAULT_CONFIG.benchmark,
) -> None:
    """Build all deterministic core CSVs without finalizing their manifest."""
    prepared_prices = prepare_prices_monthly(market_config, paths)
    price_asset_ids = sorted(prepared_prices["Asset_ID"].unique().tolist())
    _, pit_asset_ids = prepare_pit_membership(market_config, paths)
    prepare_asset_metadata(price_asset_ids, pit_asset_ids, paths)
    prepare_backtest_security_events(market_config, paths)
    prepare_sector_assignments(
        market_config,
        paths,
    )
    prepare_benchmark_monthly(benchmark_config, paths)
    atomic_write_dataframe(
        price_basis_frame("backtest"),
        paths.price_basis_csv,
        index=False,
        lineterminator="\n",
    )


__all__ = [
    "build_sector_assignment_requirements",
    "prepare_asset_metadata",
    "prepare_benchmark_monthly",
    "prepare_core_data",
    "prepare_pit_membership",
    "prepare_prices_monthly",
    "prepare_sector_assignments",
    "validate_prepared_data",
]
