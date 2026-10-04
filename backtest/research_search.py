"""Validation-only research-family search with verified atomic checkpoints."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
import json
from time import perf_counter

import numpy as np
import pandas as pd

from portfolio_core.accounting_ledger import InfeasibleRebalanceError
from portfolio_core.artifacts import file_sha256
from portfolio_core.io import atomic_write_text, atomic_write_dataframe
from portfolio_core.ranking import add_deterministic_ranks
from portfolio_core.runtime_config import apply_runtime_settings
from portfolio_core.strategies.registry import build_registered_strategy, get_strategy_definition
from portfolio_core.strategies.research_parameters import ResearchParameters
from .config import DEFAULT_CONFIG
from .engine import _BacktestAuditCollector, _run_backtest_impl
from .evaluation import calculate_return_metrics
from .research_grid import (
    expand_grid,
    frozen_identity,
    candidate_id,
    readiness_bounds,
)

_WORKER = None


def _write_json(value, path):
    atomic_write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", Path(path)
    )


def _csv(frame, path, *, index=True):
    atomic_write_dataframe(frame, Path(path), index=index)


def _load_checkpoint(directory, row, identity):
    """Verify identity, candidate ID, effective parameters and artifact hashes.

    A missing checkpoint returns None; mismatched or damaged checkpoints raise.
    """
    path = directory / "checkpoint.json"
    if not path.exists():
        return None
    checkpoint = json.loads(path.read_text())
    if (
        checkpoint["identity"] != identity
        or checkpoint["row"]["Candidate_ID"] != row["Candidate_ID"]
        or checkpoint["row"]["Parameters_JSON"] != row["Parameters_JSON"]
    ):
        raise ValueError("Checkpoint identity mismatch; refusing reuse")
    for name, digest in checkpoint["artifacts"].items():
        artifact = directory / name
        if not artifact.is_file() or file_sha256(artifact) != digest:
            raise ValueError(f"Checkpoint artifact missing or modified: {artifact}")
    # Candidate numbers are grid-relative; stable IDs may survive grid ordering.
    checkpoint["row"].update(row)
    for fold in checkpoint["folds"]:
        fold["Candidate_Number"] = row["Candidate_Number"]
    return checkpoint


def _initialize(data, identity, output, evaluation):
    global _WORKER
    apply_runtime_settings()
    _WORKER = (data, identity, Path(output), evaluation)


def _evaluate(row):
    """Return the candidate result, validation folds and whether a cache was reused.

    Insufficient history or infeasible execution leaves the candidate unranked;
    other errors propagate. Write the checkpoint after all artifacts and hashes.
    """
    data, identity, output, evaluation = _WORKER
    directory = output / "candidates" / row["Candidate_ID"]
    cached = _load_checkpoint(directory, row, identity)
    if cached is not None:
        return cached["row"], cached["folds"], True
    started = perf_counter()
    directory.mkdir(parents=True, exist_ok=True)
    packet = ResearchParameters.from_payload(json.loads(row["Requested_JSON"]))
    strategy = build_registered_strategy(packet.signal.family, packet.payload())
    collector = _BacktestAuditCollector(strategy)
    dates = pd.date_range(evaluation.initial_research_start_date, evaluation.validation_end_date, freq="ME")
    expected = pd.date_range(evaluation.validation_start_date, evaluation.validation_end_date, freq="ME")
    required_start = expected[0] - pd.offsets.MonthEnd(1)
    if not dates.isin(data.data_close.index).all():
        raise ValueError("Missing required development endpoints")
    result = {
        **row,
        "Status": "unranked_history",
        "Reason": "insufficient_validation_history",
    }
    folds = []
    artifact_names = []

    def save(name, frame, index=True):
        _csv(frame, directory / name, index=index)
        artifact_names.append(name)

    try:
        if not row["History_Unready"]:
            nav, diag, holdings = _run_backtest_impl(
                data,
                dates,
                strategy,
                return_diagnostics=True,
                return_holdings=True,
                strategy_audit=collector,
            )
            returns = nav.pct_change(fill_method=None).reindex(expected)
            if returns.isna().any():
                if nav.empty or nav.index[0] > required_start:
                    result["Reason"] = f"actual_start_too_late_for_{len(expected)}_validation_returns"
                else:
                    raise ValueError(
                        "Unexpected missing validation month after activation"
                    )
            else:
                metrics = calculate_return_metrics(
                    returns,
                    cash_interest_rate=DEFAULT_CONFIG.accounting.cash_interest_rate,
                )
                gross_collector = _BacktestAuditCollector(strategy)
                gross_nav, gross_diag = _run_backtest_impl(
                    data,
                    dates,
                    strategy,
                    apply_fees=False,
                    apply_spread=False,
                    return_diagnostics=True,
                    strategy_audit=gross_collector,
                )
                gross_returns = gross_nav.pct_change(fill_method=None).reindex(expected)
                if gross_returns.isna().any() or not np.isfinite(gross_returns).all():
                    raise ValueError(
                        "Gross-of-trading-cost run lacks complete validation"
                    )
                fold_values = []
                for year in sorted(set(expected.year)):
                    metrics_year = calculate_return_metrics(
                        returns[returns.index.year == year],
                        cash_interest_rate=DEFAULT_CONFIG.accounting.cash_interest_rate,
                    )
                    fold_values.append(metrics_year["PctReturn"])
                    folds.append(
                        dict(
                            Candidate_ID=row["Candidate_ID"],
                            Candidate_Number=row["Candidate_Number"],
                            Parameters_JSON=row["Parameters_JSON"],
                            Validation_Year=year,
                            **{"Fold_" + k: v for k, v in metrics_year.items()},
                            **{
                                "Fold_Total_"
                                + k: float(diag.loc[diag.index.year == year, k].sum())
                                for k in (
                                    "Turnover",
                                    "Fees",
                                    "Spread_Cost",
                                    "Cash_Interest_Credit",
                                    "Loan_Interest_Charge",
                                )
                            },
                        )
                    )
                validation_diag = diag.reindex(expected)
                result.update(
                    Status="ranked",
                    Reason="",
                    Validation_PctReturn=metrics["PctReturn"],
                    Validation_Sharpe=metrics["Sharpe"],
                    Maximum_Drawdown_Pct=metrics["Maximum_Drawdown_Pct"],
                    Worst_Fold_PctReturn=min(fold_values),
                    Median_Fold_PctReturn=float(np.median(fold_values)),
                    Validation_Return_Count=len(returns),
                    Actual_First_Cutoff=str(nav.index[0].date()),
                )
                for name in (
                    "Turnover",
                    "Fees",
                    "Spread_Cost",
                    "Cash_Interest_Credit",
                    "Loan_Interest_Charge",
                ):
                    result["Total_" + name] = float(validation_diag[name].sum())
                save("gross_nav.csv.gz", gross_nav.rename("NAV").to_frame())
                save("gross_diagnostics.csv.gz", gross_diag)
                save("gross_decisions.csv.gz", gross_collector.decisions_frame(), False)
                save("gross_trades.csv.gz", gross_collector.trades_frame(), False)
                save(
                    "gross_execution_constraints.csv.gz",
                    pd.DataFrame(gross_collector.research_records),
                    False,
                )
                save(
                    "gross_sector_residuals.csv.gz",
                    pd.DataFrame(gross_collector.sector_records),
                    False,
                )
                save(
                    "validation_returns.csv.gz",
                    pd.DataFrame(
                        {"Net": returns, "Gross_Of_Trading_Costs": gross_returns}
                    ),
                )
            save("net_nav.csv.gz", nav.rename("NAV").to_frame())
            save("net_diagnostics.csv.gz", diag)
            save("holdings.csv.gz", holdings, False)
    except InfeasibleRebalanceError as exc:
        result.update(Status="unranked_infeasible", Reason=str(exc))
        folds = []
    save("decisions.csv.gz", collector.decisions_frame(), False)
    save("trades.csv.gz", collector.trades_frame(), False)
    save("execution_constraints.csv.gz", pd.DataFrame(collector.research_records), False)
    save("sector_residuals.csv.gz", pd.DataFrame(collector.sector_records), False)
    _write_json(packet.payload(), directory / "requested_parameters.json")
    artifact_names.append("requested_parameters.json")
    constructed = [
        r for r in collector.research_records if r.get("Construction_Ready") == 1
    ]
    result["First_Feasible_Construction_Decision"] = (
        str(constructed[0]["Date"].date()) if constructed else None
    )
    checkpoint = dict(
        evaluation_seconds=perf_counter() - started,
        identity=identity,
        row=result,
        folds=folds,
        artifacts={name: file_sha256(directory / name) for name in artifact_names},
    )
    _write_json(checkpoint, directory / "checkpoint.json")
    return result, folds, False


def run_research_search(
    data,
    grid,
    output,
    *,
    workers=1,
    preflight_only=False,
    candidate_numbers=None,
    root=None,
    evaluation_config=DEFAULT_CONFIG.evaluation,
):
    """Search data already limited to development and return configuration metadata.

    Preflight writes grid audits, configuration and any sector-momentum history;
    it neither evaluates candidates nor validates checkpoints. Evaluation uses
    verified checkpoints, records history/infeasibility failures as unranked,
    and adds completed/ranked/resumed counts to the returned configuration.
    Other errors propagate and may leave partial files or completed checkpoints.
    """
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    started = perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(root or Path(__file__).resolve().parents[1])
    identity = frozen_identity(data, DEFAULT_CONFIG.accounting, root)
    identity.update(
        evaluation={k: str(v) for k, v in asdict(evaluation_config).items()},
        strategy_version=get_strategy_definition(grid["strategy_id"]).strategy_version,
        strategy_id=grid["strategy_id"],
    )
    if grid['strategy_id'] == 'sector_momentum':
        from dataclasses import replace
        from .sector_returns import build_sector_returns, SECTOR_RETURN_VERSION
        history = build_sector_returns(
            data, evaluation_config.validation_end_date,
            start=evaluation_config.initial_research_start_date,
        )
        data = replace(data, sector_return_history=history)
        files = {'sector_returns.csv.gz': history.returns, 'sector_return_constituents.csv.gz': history.constituents,
                 'sector_return_summary.csv.gz': history.summary}
        for name, frame in files.items():
            _csv(frame, output/name, index=name == 'sector_returns.csv.gz')
        _write_json(dict(contract=SECTOR_RETURN_VERSION,
                         start=str(evaluation_config.initial_research_start_date),
                         end=str(evaluation_config.validation_end_date),
                         identity=identity, artifacts={name:file_sha256(output/name) for name in files}),
                    output/'sector_return_manifest.json')
    rows, rejected, duplicates = expand_grid(grid)
    if not rows:
        raise ValueError("Conditional grid has no compatible candidates")
    bounds = {}
    for row in rows:
        packet = ResearchParameters.from_payload(json.loads(row["Requested_JSON"]))
        key = (
            packet.signal,
            packet.selection,
            packet.sizing,
        )
        if key not in bounds:
            bounds[key] = readiness_bounds(packet, data, evaluation_config=evaluation_config)
        row.update(bounds[key], Candidate_ID=candidate_id(row, identity))
    _csv(pd.DataFrame(rows), output / "preflight.csv.gz", index=False)
    _csv(pd.DataFrame(rejected), output / "rejected.csv.gz", index=False)
    _csv(pd.DataFrame(duplicates), output / "duplicates.csv.gz", index=False)
    selected = set(
        candidate_numbers
        if candidate_numbers is not None
        else (r["Candidate_Number"] for r in rows)
    )
    if not selected.issubset({r["Candidate_Number"] for r in rows}):
        raise ValueError("Unknown candidate number")
    work = [r for r in rows if r["Candidate_Number"] in selected]
    if not work and not preflight_only:
        raise ValueError("No candidates selected for evaluation")
    configuration = dict(
        identity=identity,
        grid=grid,
        valid_candidates=len(rows),
        rejected_requests=len(rejected),
        duplicate_requests=len(duplicates),
        history_bound_unready=sum(r["History_Unready"] for r in rows),
        mode=(
            "preflight"
            if preflight_only
            else "selected" if candidate_numbers is not None else "full"
        ),
        selected_candidate_numbers=sorted(selected),
        workers=workers,
        validation_returns=len(pd.date_range(
            evaluation_config.validation_start_date, evaluation_config.validation_end_date, freq="ME",
        )),
        final_holdout_excluded=True,
    )
    _write_json(configuration, output / "search_configuration.json")
    print(
        f"Preflight: {len(rows)} valid, {len(rejected)} rejected, {len(duplicates)} duplicate requests; {configuration['history_bound_unready']} history-bound unready.",
        flush=True,
    )
    if preflight_only:
        return configuration
    preflight_seconds = perf_counter() - started
    evaluation_started = perf_counter()
    results, folds, resumed = [], [], 0

    def collect(outcome):
        nonlocal resumed
        row, candidate_folds, cached = outcome
        results.append(row)
        folds.extend(candidate_folds)
        resumed += int(cached)
        print(
            f"[{len(results)}/{len(work)}] candidate {row['Candidate_Number']}: {row['Status']}"
            + (" (resumed)" if cached else ""),
            flush=True,
        )

    if workers == 1:
        _initialize(data, identity, output, evaluation_config)
        for row in work:
            collect(_evaluate(row))
    else:
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_initialize,
            initargs=(data, identity, output, evaluation_config),
        )
        try:
            futures = {executor.submit(_evaluate, row): row for row in work}
            for future in as_completed(futures):
                try:
                    collect(future.result())
                except BrokenProcessPool as exc:
                    row = futures[future]
                    raise RuntimeError(
                        f"Search worker stopped while awaiting {grid['strategy_id']} "
                        f"candidate {row['Candidate_Number']}: {exc}"
                    ) from exc
        except BaseException:
            executor.terminate_workers()
            raise
        else:
            executor.shutdown(wait=True)
    frame = pd.DataFrame(results).sort_values("Candidate_Number").reset_index(drop=True)
    ranked = frame.Status.eq("ranked")
    frame["PctReturn_Rank"] = np.nan
    frame["Sharpe_Rank"] = np.nan
    if ranked.any():
        ranks = add_deterministic_ranks(
            frame.loc[ranked],
            (
                ("Validation_PctReturn", "PctReturn_Rank"),
                ("Validation_Sharpe", "Sharpe_Rank"),
            ),
            tie_breaker="Parameters_JSON",
        )
        frame.loc[ranked, ["PctReturn_Rank", "Sharpe_Rank"]] = ranks[
            ["PctReturn_Rank", "Sharpe_Rank"]
        ]
    _csv(
        frame.reindex(sorted(frame.columns), axis=1),
        output / "candidates.csv.gz",
        index=False,
    )
    fold_frame = pd.DataFrame(folds)
    if not fold_frame.empty:
        fold_frame = fold_frame.sort_values(
            ["Candidate_Number", "Validation_Year"], kind="stable",
        ).reset_index(drop=True)
    _csv(
        fold_frame.reindex(sorted(fold_frame.columns), axis=1),
        output / "fold_metrics.csv.gz",
        index=False,
    )
    outcomes = {r["Candidate_ID"]: r for r in results}
    coverage = []
    for row in rows:
        outcome = outcomes.get(row["Candidate_ID"], {})
        coverage.append(
            dict(
                Candidate_Number=row["Candidate_Number"],
                Candidate_ID=row["Candidate_ID"],
                Status=outcome.get(
                    "Status",
                    (
                        "unranked_history_bound"
                        if row["History_Unready"]
                        else "not_evaluated"
                    ),
                ),
                Reason=outcome.get(
                    "Reason",
                    (
                        "history_bound_exceeds_validation_start"
                        if row["History_Unready"]
                        else "outside_selected_run"
                    ),
                ),
            )
        )
    _csv(pd.DataFrame(coverage), output / "coverage.csv.gz", index=False)
    configuration.update(
        completed_candidates=len(results),
        ranked_candidates=int(ranked.sum()),
        resumed_candidates=resumed,
    )
    _write_json(configuration, output / "search_configuration.json")
    # Keep nondeterministic timing outside deterministic result/fold tables.
    _write_json(dict(
        clock="time.perf_counter",
        preflight_seconds=preflight_seconds,
        evaluation_wall_seconds=perf_counter() - evaluation_started,
        total_wall_seconds=perf_counter() - started,
        workers=workers,
        resumed_candidates=resumed,
        candidates=[dict(
            Candidate_Number=r["Candidate_Number"], Status=r["Status"],
            evaluation_seconds=json.loads((output / "candidates" /
                r["Candidate_ID"] / "checkpoint.json").read_text()).get("evaluation_seconds"),
        ) for r in sorted(results, key=lambda r: r["Candidate_Number"])],
    ), output / "timing.json")
    return configuration
