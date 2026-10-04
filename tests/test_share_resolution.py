"""Independent edge cases for the reviewed retrospective shares pipeline."""
from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest.config import DEFAULT_CONFIG
from backtest import acquisition_planning, acquisition_execution, shares_preparation
from backtest.share_resolution import (
    ReviewedShares, ShareResolver, required_pairs, pairs_digest,
    validate_prepared_shares,
)
from portfolio_core.shares import CanonicalShareIdentity, RAW_SHARES_COLUMNS
from portfolio_core.brinson_attribution import build_brinson_benchmark, BENCHMARK_CONSTITUENT_COLUMNS


def observation(day="2024-01-01", shares=100, **kwargs):
    return dict(Observation_ID="fact-"+day, Asset_ID="A", CIK="0000000001",
                Observation_Date=day, Shares=str(shares), Filed=day, Kind="sec",
                Status="selected", Post_Event="False", Base_Observation_Date="",
                Source_URL="https://www.sec.gov/example", Class_Tier="reviewed common",
                Review_Notes="Explicit outstanding count", Base_Shares="", Issued_Shares="", **kwargs)


def event(day="2024-02-01", action="multiply", factor=2, **kwargs):
    row = dict(Asset_ID="A", Event_ID="event-"+day, Effective_Date=day,
               Estimator_Action=action, Estimator_Share_Multiplier=str(factor),
               Share_Continuity_Effect="", CIK_Before="", CIK_After="",
               Earliest_Effective_Date="", Latest_Effective_Date="",
               Source_URLs="https://www.sec.gov/event", Review_Claim="Documented event terms")
    row.update(kwargs)
    return row


def bundle(observations=None, events=None, issuers=None, factors=None):
    return ReviewedShares(
        pd.DataFrame(observations if observations is not None else [observation()]),
        pd.DataFrame(issuers if issuers is not None else [dict(
            Asset_ID="A", Effective_Start="2020-01-01", Effective_End="2030-01-01",
            Expected_CIK="0000000001", Source_URLs="https://www.sec.gov/issuer")]),
        pd.DataFrame(events or [], columns=list(event())),
        pd.DataFrame(factors or [], columns=["Interval_ID", "Asset_ID", "Effective_Start",
                     "Effective_End", "Factor", "Source_URLs", "Review_Notes"]), {},
    )


def pairs(*dates):
    return pd.DataFrame([(pd.Timestamp(day), "A") for day in dates], columns=["Date", "Asset_ID"])


def yahoo(values, day="2024-05-01", symbol="A", start="", end=""):
    return pd.DataFrame([["A", symbol, start, end, day, i, value] for i, value in enumerate(values)],
                        columns=RAW_SHARES_COLUMNS)


def resolve(reviewed, dates=("2024-05-31",), raw=None, identities=None):
    return ShareResolver(reviewed).resolve(pairs(*dates), raw,
        identities if identities is not None else (CanonicalShareIdentity("A", "A"),))


def test_source_priority_and_retrospective_publication():
    reported = observation(); reported["Filed"] = "2024-03-01"
    estimate = observation("2024-01-20", 150); estimate.update(Kind="event_estimate", Post_Event="True")
    selected, missing = resolve(bundle([reported, estimate]), ("2024-01-31",), yahoo([170], "2024-01-25"))
    assert missing.empty
    assert selected.Source.tolist() == ["sec"]
    assert selected.Shares_Outstanding.tolist() == [100]
    assert selected.Filed.tolist() == ["2024-03-01"]


@pytest.mark.parametrize("age,covered", [(0, True), (120, True), (121, False), (-1, False)])
def test_reported_age_uses_observation_date(age, covered):
    day = pd.Timestamp("2024-01-01") + pd.Timedelta(days=age)
    selected, missing = resolve(bundle(), (str(day.date()),))
    assert bool(len(selected)) is covered
    assert bool(len(missing)) is not covered


@pytest.mark.parametrize("status", ["blocked", "conflicting_values"])
def test_latest_block_cannot_be_evaded_with_older_report(status):
    latest = observation("2024-01-15", 110); latest["Status"] = status
    selected, missing = resolve(bundle([observation(), latest]), ("2024-01-31",))
    assert selected.empty
    assert status in missing.Reason.iloc[0]


def test_event_estimate_has_separate_priority_and_does_not_reset_base_age():
    estimate = observation("2024-05-01", 130)
    estimate.update(Kind="event_estimate", Base_Observation_Date="2024-01-01", Post_Event="True")
    selected, missing = resolve(bundle([observation(), estimate]), ("2024-05-31",))
    assert selected.empty
    assert "stale_event_base" in missing.Reason.iloc[0]
    estimate["Base_Observation_Date"] = "2024-03-01"
    selected, missing = resolve(bundle([observation(), estimate]))
    assert selected.Source.tolist() == ["event_estimate"]
    assert selected.Shares_Outstanding.tolist() == [130]


@pytest.mark.parametrize("values,expected,sequences", [
    ([100, 180], 100, "0"), ([180, 100], 100, "1"),
    ([100, 100, 180], 100, "0;1"), ([100, 102], None, ""),
    ([170, 180], None, ""), ([95, 100, 105], None, ""),
])
def test_all_yahoo_sequences_require_one_distinct_compatible_value(values, expected, sequences):
    selected, missing = resolve(bundle(), raw=yahoo(values))
    if expected is None:
        assert selected.empty
        assert "compatible_values" in missing.Reason.iloc[0]
    else:
        assert missing.empty
        assert selected.Source.tolist() == ["yahoo"]
        assert selected.Shares_Outstanding.tolist() == [expected]
        assert selected.Observation_Sequences.tolist() == [sequences]
        assert selected.Reference_Date.tolist() == ["2024-01-01"]


def test_ambiguous_latest_yahoo_response_does_not_revive_older_response():
    raw = pd.concat([yahoo([100], "2024-04-20"), yahoo([100, 102])], ignore_index=True)
    selected, missing = resolve(bundle(), raw=raw)
    assert selected.empty
    assert "compatible_values:2" in missing.Reason.iloc[0]


@pytest.mark.parametrize("anchor_day,covered", [("2023-05-02", True), ("2023-05-01", False)])
def test_reference_age_boundary(anchor_day, covered):
    selected, _ = resolve(bundle([observation(anchor_day)]), raw=yahoo([100]))
    assert bool(len(selected)) is covered


def test_future_reference_cannot_support_yahoo():
    # Latest reported row is stale at target but still follows Yahoo's date.
    selected, missing = resolve(bundle([observation("2024-05-02")]),
                                ("2024-09-01",), yahoo([100], "2024-05-01"))
    assert selected.empty
    assert "stale_yahoo" in missing.Reason.iloc[0]
    resolver = ShareResolver(bundle([observation("2024-05-02")]))
    result, reason = resolver.yahoo("A", "2024-05-31", yahoo([100]).to_dict("records"))
    assert result is None and reason == "no_qualified_reference"


def test_split_applies_once_and_never_on_observation_day():
    duplicate = event(); duplicate["Event_ID"] = "second-source-same-split"
    selected, _ = resolve(bundle(events=[event(), duplicate]), ("2024-01-31", "2024-02-29"))
    assert selected.Shares_Outstanding.tolist() == [100, 200]
    assert selected.Share_Factor.tolist() == [1, 2]
    selected, _ = resolve(bundle([observation("2024-02-01", 200)], [event()]), ("2024-02-29",))
    assert selected.Shares_Outstanding.tolist() == [200]


def test_yahoo_reference_is_split_adjusted_before_consistency_screen():
    selected, missing = resolve(bundle(events=[event()]), raw=yahoo([100, 200]))
    assert missing.empty
    assert selected.Shares_Outstanding.tolist() == [200]
    assert selected.Observation_Sequences.tolist() == ["1"]


@pytest.mark.parametrize("post_event,covered", [("True", True), ("False", False)])
def test_same_day_capital_event_requires_explicit_post_event_count(post_event, covered):
    obs = observation("2024-02-01", 200); obs["Post_Event"] = post_event
    selected, _ = resolve(bundle([obs], [event(action="new_observation_required")]), ("2024-02-29",))
    assert bool(len(selected)) is covered


def test_capital_event_blocks_prior_counts_and_yahoo_anchors():
    reviewed = bundle(events=[event(action="new_observation_required")])
    selected, missing = resolve(reviewed, ("2024-02-29",), yahoo([100], "2024-02-15"))
    assert selected.empty
    assert "event_block" in missing.Reason.iloc[0]
    assert "no_qualified_reference" in missing.Reason.iloc[0]


def test_uncertain_event_blocks_entire_uncertainty_interval():
    uncertain = event(action="uncertain", Earliest_Effective_Date="2024-02-01",
                      Latest_Effective_Date="2024-02-29")
    selected, missing = resolve(bundle(events=[uncertain]), ("2024-01-31", "2024-02-29"))
    assert len(selected) == len(missing) == 1
    assert selected.Date.iloc[0] == pd.Timestamp("2024-01-31")


def test_issuer_transition_requires_documented_continuity():
    issuers = [dict(Asset_ID="A", Effective_Start="2020-01-01", Effective_End="2024-02-01", Expected_CIK="0000000001"),
               dict(Asset_ID="A", Effective_Start="2024-02-01", Effective_End="2030-01-01", Expected_CIK="0000000002")]
    reviewed = bundle(issuers=issuers)
    selected, _ = resolve(reviewed, ("2024-02-29",))
    assert selected.empty
    continuity = event(action="continue", CIK_Before="0000000001", CIK_After="0000000002",
                       Share_Continuity_Effect="documented_same_share_unit")
    selected, _ = resolve(bundle(issuers=issuers, events=[continuity]), ("2024-02-29",))
    assert selected.Shares_Outstanding.tolist() == [100]


def test_wrong_issuer_observation_is_never_selected():
    obs = observation(); obs["CIK"] = "0000000002"
    selected, _ = resolve(bundle([obs]), ("2024-01-31",))
    assert selected.empty


def test_yahoo_requires_full_provider_identity_and_effective_interval():
    wrong = yahoo([100], symbol="OLD")
    selected, _ = resolve(bundle(), raw=wrong)
    assert selected.empty
    raw = yahoo([100], end="2024-05-15")
    selected, _ = resolve(bundle(), raw=raw, identities=(CanonicalShareIdentity("A", "A", "", "2024-05-15"),))
    assert selected.empty


def test_overlapping_yahoo_identities_are_unresolved_even_with_equal_values():
    raw = pd.concat([yahoo([100]), yahoo([100], symbol="OTHER")], ignore_index=True)
    selected, missing = resolve(bundle(), raw=raw, identities=(CanonicalShareIdentity("A", "A"), CanonicalShareIdentity("A", "OTHER")))
    assert selected.empty
    assert "overlapping_yahoo_identities" in missing.Reason.iloc[0]


def test_primary_coverage_needs_no_yahoo_input_or_identity():
    selected, missing = resolve(bundle(), ("2024-01-31",), pd.DataFrame({"corrupt": [1]}), identities=())
    assert len(selected) == 1 and missing.empty


def test_factors_do_not_modify_nominal_counts():
    factors = [dict(Interval_ID="split-price", Asset_ID="A", Effective_Start="2024-01-01",
                    Effective_End="2024-01-31", Factor=10, Source_URLs="source", Review_Notes="split basis")]
    selected, _ = resolve(bundle(factors=factors), ("2024-01-31", "2024-02-29"))
    assert selected.Shares_Outstanding.tolist() == [100, 100]
    assert selected.Capitalization_Factor.tolist() == [10, 1]


def write_bundle(tmp_path, reviewed):
    tmp_path.mkdir(exist_ok=True)
    for name, frame in [("observations.csv", reviewed.observations), ("issuer_intervals.csv", reviewed.issuers),
                        ("events.csv", reviewed.events), ("capitalization_price_factors.csv", reviewed.factors)]:
        frame.to_csv(tmp_path / name, index=False)
    prices, basis = tmp_path / "prices", tmp_path / "basis"
    prices.write_text("fixed prices"); basis.write_text("fixed basis")
    metadata = dict(schema_version=1, reviewed_start_date="2024-01-31", reviewed_end_date="2024-01-31",
                    required_pairs_sha256=pairs_digest(pairs("2024-01-31")),
                    prepared_prices_sha256=hashlib.sha256(prices.read_bytes()).hexdigest(),
                    price_basis_sha256=hashlib.sha256(basis.read_bytes()).hexdigest())
    (tmp_path / "review.json").write_text(json.dumps(metadata))
    return prices, basis


def test_review_metadata_binds_fixed_inputs_and_required_pairs(tmp_path):
    prices, basis = write_bundle(tmp_path, bundle())
    reviewed = ReviewedShares.load(tmp_path)
    reviewed.validate_inputs(pairs("2024-01-31"), prices, basis)
    with pytest.raises(ValueError, match="population"):
        reviewed.validate_inputs(pairs("2024-02-29"), prices, basis)
    prices.write_text("changed")
    with pytest.raises(ValueError, match="Price inputs changed"):
        reviewed.validate_inputs(pairs("2024-01-31"), prices, basis)


@pytest.mark.parametrize("defect", ["duplicate", "negative", "status", "notes", "issuer_overlap", "factor_overlap"])
def test_reviewed_bundle_rejects_ambiguous_or_invalid_decisions(tmp_path, defect):
    reviewed = bundle()
    if defect == "duplicate":
        reviewed = replace(reviewed, observations=pd.concat([reviewed.observations]*2))
    elif defect == "negative":
        reviewed.observations.loc[0, "Shares"] = "-1"
    elif defect == "status":
        reviewed.observations.loc[0, "Status"] = "guess"
    elif defect == "notes":
        reviewed.observations.loc[0, "Review_Notes"] = ""
    elif defect == "issuer_overlap":
        reviewed = replace(reviewed, issuers=pd.concat([reviewed.issuers]*2))
    else:
        row = dict(Interval_ID="one", Asset_ID="A", Effective_Start="2024-01-01", Effective_End="2024-02-01",
                   Factor="2", Source_URLs="source", Review_Notes="basis")
        reviewed = replace(reviewed, factors=pd.DataFrame([row, {**row, "Interval_ID": "two"}]))
    write_bundle(tmp_path, reviewed)
    with pytest.raises(ValueError):
        ReviewedShares.load(tmp_path)


@pytest.mark.parametrize("defect", ["wide", "duplicate", "negative", "age", "factor", "source"])
def test_prepared_schema_fails_closed(defect):
    selected, _ = resolve(bundle(), ("2024-01-31",))
    if defect == "wide":
        selected = selected.pivot(index="Date", columns="Asset_ID", values="Shares_Outstanding")
    elif defect == "duplicate":
        selected = pd.concat([selected]*2, ignore_index=True)
    else:
        column, value = {"negative": ("Shares_Outstanding", -1), "age": ("Age_Days", 400),
                         "factor": ("Capitalization_Factor", 0), "source": ("Source", "guess")}[defect]
        selected.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_prepared_shares(selected)


def test_capitalization_changes_weights_but_not_stock_returns_and_default_is_unchanged():
    constituents = pd.DataFrame([
        ["p", "2024-01-31", "2024-02-29", "A", "10", "Energy", 10, 11, 100],
        ["p", "2024-01-31", "2024-02-29", "B", "20", "Industrials", 10, 9, 100],
    ], columns=BENCHMARK_CONSTITUENT_COLUMNS)
    default, default_audit = build_brinson_benchmark(constituents)
    unit, unit_audit = build_brinson_benchmark(constituents, capitalization_factors=pd.Series([1, 1]))
    pd.testing.assert_frame_equal(default, unit)
    pd.testing.assert_frame_equal(default_audit, unit_audit)
    adjusted, audit = build_brinson_benchmark(constituents, capitalization_factors=pd.Series([2, 1]))
    assert adjusted.Benchmark_Weight.tolist() == pytest.approx([2/3, 1/3])
    assert adjusted.Benchmark_Return.tolist() == pytest.approx(default.Benchmark_Return)
    assert audit.Benchmark_Market_Cap.iloc[0] == 3000
    for factors in [pd.Series([1, 1], index=[1, 0]), pd.Series([0, 1]), pd.Series([np.inf, 1])]:
        with pytest.raises(ValueError, match="Capitalization factors"):
            build_brinson_benchmark(constituents, capitalization_factors=factors)


def test_preparation_is_offline_deterministic_and_validates_before_saving(tmp_path, monkeypatch):
    from backtest.paths import BacktestSharesPaths
    membership = pd.DataFrame(True, index=pd.DatetimeIndex(["2024-01-31"]), columns=["A"])
    resolver = ShareResolver(bundle())
    selected, missing = resolver.resolve(required_pairs(membership))
    plan = SimpleNamespace(resolver=resolver, pairs=required_pairs(membership), canonical_identities=())
    monkeypatch.setattr(acquisition_planning, "plan_share_sources", lambda *a, **k: plan)
    monkeypatch.setattr(shares_preparation, "load_raw_shares", lambda *a: pytest.fail("primary needs no Yahoo file"))
    market = SimpleNamespace(data_close=membership.astype(float), pit_matrix=membership,
                             asset_to_ticker={"A": "A"}, unavailable_members=())
    paths = BacktestSharesPaths(
        tmp_path / "raw", tmp_path / "prepared", tmp_path / "raw/provenance/shares",
    )
    config = SimpleNamespace(paths=SimpleNamespace(shares=paths), brinson=replace(DEFAULT_CONFIG.brinson,
                             start_date=pd.Timestamp("2024-01-31").date(), end_date=pd.Timestamp("2024-01-31").date()))
    shares_preparation.prepare_shares_dataset(market, config)
    before = paths.final_shares_csv.read_bytes()
    shares_preparation.prepare_shares_dataset(market, config)
    assert paths.final_shares_csv.read_bytes() == before
    loaded = shares_preparation.load_prepared_shares(paths, config.brinson)
    assert loaded.Shares_Outstanding.tolist() == [100]
    plan.resolver = ShareResolver(bundle([observation("2025-01-01")]))
    with pytest.raises(RuntimeError, match="Unresolved"):
        shares_preparation.prepare_shares_dataset(market, config)
    assert paths.final_shares_csv.read_bytes() == before


def test_materialized_resolution_matches_reviewed_feasibility():
    market = acquisition_planning.load_fresh_core_data()
    plan = acquisition_planning.plan_share_sources(market)
    from portfolio_core.shares import load_raw_shares
    selected, missing = plan.resolver.resolve(plan.pairs, load_raw_shares(DEFAULT_CONFIG.paths.shares.raw_shares_csv),
                                             plan.canonical_identities)
    assert missing.empty and len(selected) == 12576
    assert selected.Source.value_counts().to_dict() == {"sec": 12501, "yahoo": 56, "event_estimate": 19}
    assert selected.Capitalization_Factor.ne(1).sum() == 476
    numeric = selected.rename(columns={"Shares_Outstanding": "Shares"})[["Date", "Asset_ID", "Shares", "Source"]]
    assert hashlib.sha256(numeric.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode()).hexdigest() == "ea824c0f8163a30b4ff9fa303ed47cba0d3812add565f811a19a4877f41a76f7"


def test_primary_only_plan_never_resolves_yahoo_identity(tmp_path, monkeypatch):
    from backtest.paths import BacktestSharesPaths
    directory = tmp_path / "raw" / "provenance" / "shares"
    directory.parent.mkdir(parents=True)
    prices, basis = write_bundle(directory, bundle())
    membership = pd.DataFrame(True, index=pd.DatetimeIndex(["2024-01-31"]), columns=["A"])
    market = SimpleNamespace(data_close=membership.astype(float), pit_matrix=membership,
                             asset_to_ticker={"A": "A"})
    config = SimpleNamespace(paths=SimpleNamespace(
        shares=BacktestSharesPaths(tmp_path / "raw/shares", tmp_path / "prepared", directory),
        prices_monthly_csv=prices, price_basis_csv=basis), brinson=replace(DEFAULT_CONFIG.brinson,
        start_date=pd.Timestamp("2024-01-31").date(), end_date=pd.Timestamp("2024-01-31").date()))
    monkeypatch.setattr(acquisition_planning, "_build_yahoo_shares_identities",
                        lambda *a, **k: pytest.fail("primary coverage must precede Yahoo identity lookup"))
    plan = acquisition_planning.plan_share_sources(market, config=config)
    assert plan.yahoo_identities == ()
    assert len(plan.primary) == 1
    assert acquisition_planning.build_shares_requests(plan) == []
    records, detail = acquisition_planning.shares_readiness_records(plan, pd.DataFrame({"bad": [1]}))
    assert records[0].covered_count == 1 and detail.Covered.all()


@pytest.mark.parametrize("refresh", [False, True])
def test_acquisition_initializes_provider_only_for_actual_requests_and_retains_failed_refresh(tmp_path, monkeypatch, refresh):
    from backtest.paths import BacktestSharesPaths
    from data_acquisition.contracts import AcquisitionIdentity, AcquisitionRequest, AcquisitionStatus, ProviderStatus
    from data_acquisition.engine import AcquisitionPolicy
    from data_acquisition.providers.base import ProviderAdapter
    from data_acquisition.errors import RetryableProviderError
    from contextlib import nullcontext
    from data_acquisition.contracts import ReadinessRecord, ReadinessStatus
    path = BacktestSharesPaths(
        tmp_path / "raw", tmp_path / "prepared", tmp_path / "raw/provenance/shares",
    )
    path.raw_dir.mkdir()
    raw = yahoo([100])
    request = AcquisitionRequest(AcquisitionIdentity(scope="backtest", dataset="shares", asset_id="A",
        provider="yahoo", provider_symbol="A"), requested_start="2023-01-01", requested_end="2024-06-01")
    status = AcquisitionStatus(identity=request.identity, status=ProviderStatus.OK,
        requested_start=request.requested_start, requested_end=request.requested_end,
        observation_count=1, observation_start="2024-05-01", observation_end="2024-05-01")
    monkeypatch.setattr(acquisition_execution, "DEFAULT_CONFIG", SimpleNamespace(paths=SimpleNamespace(shares=path)))
    monkeypatch.setattr(acquisition_execution, "load_fresh_core_data", lambda *a: object())
    monkeypatch.setattr(acquisition_execution, "plan_share_sources", lambda *a, **k: SimpleNamespace(yahoo_identities=(object(),)))
    monkeypatch.setattr(acquisition_execution, "build_shares_requests", lambda *a, **k: [request])
    monkeypatch.setattr(acquisition_execution, "_read_shares_state", lambda: (raw, [status]))
    monkeypatch.setattr(acquisition_execution, "acquisition_lock", lambda *a: nullcontext())
    monkeypatch.setattr(acquisition_execution, "_write_shares_manifest", lambda *a: None)
    attempts = []
    def prepare(*args):
        attempts.append("client")
        return SimpleNamespace(name="mock", version="1")
    def fetch(request):
        raise RetryableProviderError("synthetic failed refresh")
    monkeypatch.setattr(acquisition_execution, "prepare_yfinance", prepare)
    monkeypatch.setattr(acquisition_execution, "make_yahoo_shares_adapter", lambda client: ProviderAdapter(fetch, provider="yahoo", dataset="shares", client_name=client.name, client_version=client.version))
    monkeypatch.setattr(acquisition_execution, "PRODUCTION_ACQUISITION_POLICY",
                        AcquisitionPolicy(max_attempts=1))
    def readiness(market, retained, **kwargs):
        selected, missing = resolve(bundle(), raw=retained)
        assert len(selected) == 1 and missing.empty
        if refresh:
            from data_acquisition.contracts import read_acquisition_statuses
            assert read_acquisition_statuses(path.acquisition_status_csv)[0].status is ProviderStatus.FAILED
        return [ReadinessRecord(scope="backtest", dataset="shares", asset_id="*", requirement_set="test",
            required_count=1, covered_count=1, status=ReadinessStatus.COMPLETE,
            contributing_sources=("yahoo",), checked_at_utc="2026-01-01T00:00:00Z")], selected
    monkeypatch.setattr(acquisition_execution, "shares_readiness_records", readiness)
    command = SimpleNamespace(tickers_file=None, refresh=refresh, dry_run=False, project_root=tmp_path)
    assert acquisition_execution.acquire_backtest_shares(command) == (1 if refresh else 0)
    assert attempts == (["client"] if refresh else [])


def test_core_preflight_allows_reviewed_changes_but_full_brinson_check_rejects_them(monkeypatch):
    import portfolio_core.artifacts as artifacts
    from backtest.preparation_artifacts import validate_preparation_manifest
    digest = artifacts.artifact_sha256
    monkeypatch.setattr(artifacts, "artifact_sha256", lambda spec:
                        "changed" if spec.artifact == "brinson_reviewed" else digest(spec))
    validate_preparation_manifest(include_brinson=False, check_raw=True)
    with pytest.raises(RuntimeError, match="stale or modified"):
        validate_preparation_manifest(include_brinson=True, check_raw=True)


def test_analysis_manifest_check_never_reads_reviewed_or_yahoo_raw_inputs(monkeypatch):
    import portfolio_core.artifacts as artifacts
    from backtest.preparation_artifacts import validate_preparation_manifest
    digest = artifacts.artifact_sha256
    def only_prepared(spec):
        if spec.artifact in {"brinson_reviewed", "brinson_yahoo_shares"}:
            pytest.fail("Analysis read raw shares")
        return digest(spec)
    monkeypatch.setattr(artifacts, "artifact_sha256", only_prepared)
    validate_preparation_manifest(include_brinson=True, check_raw=False)


def test_zero_yahoo_plan_needs_no_checkpoint_or_provider(tmp_path, monkeypatch):
    from backtest.paths import BacktestSharesPaths
    from data_acquisition.contracts import ReadinessRecord, ReadinessStatus
    from contextlib import nullcontext
    paths = BacktestSharesPaths(
        tmp_path / "raw", tmp_path / "prepared", tmp_path / "raw/provenance/shares",
    )
    monkeypatch.setattr(acquisition_execution, "DEFAULT_CONFIG", SimpleNamespace(paths=SimpleNamespace(shares=paths)))
    monkeypatch.setattr(acquisition_execution, "load_fresh_core_data", lambda *a: object())
    monkeypatch.setattr(acquisition_execution, "build_shares_requests", lambda *a, **k: [])
    monkeypatch.setattr(acquisition_execution, "plan_share_sources", lambda *a, **k: SimpleNamespace(yahoo_identities=()))
    monkeypatch.setattr(acquisition_execution, "_read_shares_state", lambda: pytest.fail("No Yahoo dependency"))
    monkeypatch.setattr(acquisition_execution, "prepare_yfinance", lambda *a: pytest.fail("No Yahoo request"))
    monkeypatch.setattr(acquisition_execution, "acquisition_lock", lambda *a: nullcontext())
    monkeypatch.setattr(acquisition_execution, "_write_shares_manifest", lambda *a: None)
    monkeypatch.setattr(acquisition_execution, "shares_readiness_records", lambda *a, **k: ([ReadinessRecord(
        scope="backtest", dataset="shares", asset_id="*", requirement_set="test", required_count=1,
        covered_count=1, status=ReadinessStatus.COMPLETE, contributing_sources=("sec",),
        checked_at_utc="2026-01-01T00:00:00Z")], pd.DataFrame()))
    command = SimpleNamespace(tickers_file=None, refresh=True, dry_run=False, project_root=tmp_path)
    assert acquisition_execution.acquire_backtest_shares(command) == 0
