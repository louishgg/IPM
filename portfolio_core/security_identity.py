"""Canonical, scope-neutral security identity and corporate-action evidence.

The canonical tables separate four claims that were previously combined in a
single backtest CSV:

* an official security event;
* the structured legs which an accounting engine may execute;
* the exact claim supported by each official source; and
* a provider-specific symbol observed in a local artifact.

Documentary approval is deliberately distinct from accounting executability.
Free-text transaction terms are preserved for migration and review, but are
never interpreted by this module as portfolio economics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import pandas as pd

from portfolio_core.artifacts import (
    ArtifactOrigin,
    validate_exact_manifest_catalog,
)


SECURITY_IDENTITY_RELATIVE_DIR = Path("data/shared/provenance/security_identity")
UNAVAILABLE_PREFIX = "FJA_UNAVAILABLE::"

SECURITY_EVENTS_COLUMNS = (
    "Event_ID",
    "Effective_Date",
    "Event_Type",
    "Continuity_Class",
    "Accounting_Status",
    "Legacy_Event_Type",
    "Review_Status",
    "Summary",
)

SECURITY_EVENT_LEGS_COLUMNS = (
    "Event_Leg_ID",
    "Event_ID",
    "Source_ID",
    "Leg_Sequence",
    "From_Ticker",
    "To_Ticker",
    "Leg_Type",
    "Share_Ratio",
    "Executable_Share_Ratio",
    "Normalization_Denominator",
    "Normalization_Provider",
    "Normalization_Provider_Symbol",
    "Normalization_Treatment_Date",
    "Normalization_Source_ID",
    "Cash_Amount",
    "Currency",
    "CVR_Units",
    "CVR_Base_Value_Per_Unit",
    "CVR_Max_Value_Per_Unit",
    "Retain_Predecessor",
    "Tradable",
    "Terms_Text",
    "Review_Status",
    "Notes",
)

PROVIDER_ROW_EQUIVALENCE_COLUMNS = (
    "Equivalence_ID",
    "Scope",
    "Provider",
    "Asset_ID",
    "Predecessor_Symbol",
    "Successor_Symbol",
    "Effective_Date",
    "Compared_Fields",
    "Absolute_Tolerance",
    "Relative_Tolerance",
    "Common_Start",
    "Common_End",
    "Common_Row_Count",
    "Predecessor_Artifact_SHA256",
    "Successor_Response_SHA256",
    "Review_Status",
    "Review_Result",
    "Notes",
)

SECURITY_EVENT_SOURCES_COLUMNS = (
    "Source_ID",
    "Event_ID",
    "Source_Role",
    "Publisher",
    "Document_Date",
    "Source_URL",
    "Evidence_Claim",
    "Review_Status",
)

PROVIDER_SYMBOL_MAPPINGS_COLUMNS = (
    "Mapping_ID",
    "Scope",
    "Provider",
    "Source_Ticker",
    "Provider_Symbol",
    "Asset_ID",
    "Effective_Start",
    "Effective_End",
    "Resolution_Method",
    "Event_ID",
    "Primary_Source_ID",
    "Legacy_Event_Type",
    "Legacy_From_Ticker",
    "Legacy_To_Ticker",
    "Legacy_Conversion_Terms",
    "Issuer_CIK_Before",
    "Issuer_CIK_After",
    "Security_Identifier",
    "Local_First_Date",
    "Local_Last_Date",
    "Simultaneous_Symbol_Conflict",
    "Provider_Evidence_Status",
    "Provider_Source_URL",
    "Provider_Evidence_Claim",
    "Review_Status",
    "Notes",
)

ALLOWED_EVENT_TYPES = frozenset({
    "identity_continuity",
    "otc_transition",
    "cash_settlement",
    "stock_exchange",
    "cash_and_stock",
    "distribution",
    "cancellation",
})
ALLOWED_CONTINUITY_CLASSES = frozenset({
    "same_security",
    "predecessor_extinguished",
    "predecessor_survives",
    "cancelled",
})
ALLOWED_ACCOUNTING_STATUSES = frozenset({
    "executable",
    "documented_not_executable",
})
ALLOWED_LEG_TYPES = frozenset({
    "relabel",
    "cash",
    "stock",
    "distribution",
    "cvr",
    "cancellation",
})
_SUPPORTED_NORMALIZATION_PROVIDERS = frozenset({"reuters", "yahoo"})
ALLOWED_SOURCE_ROLES = frozenset({"primary", "terms", "supplemental"})
ALLOWED_REVIEW_STATUSES = frozenset({"approved", "unresolved"})
ALLOWED_EQUIVALENCE_RESULTS = frozenset({
    "equivalent",
    "rejected",
    "not_evaluated",
})
ALLOWED_RESOLUTION_METHODS = frozenset({
    "official_same_security_local_ric",
    "official_successor_local_ric",
    "official_otc_transition_local_ric",
    "reviewed_missing_price",
    "reviewed_wikipedia_mapping",
    "reviewed_yahoo_alias",
    "reviewed_effective_symbol",
    "reviewed_yahoo_close_fallback",
    "reviewed_yahoo_event_treatment",
    "reviewed_wiki_close_fallback",
    "reviewed_local_reuters_identity",
})

_OFFICIAL_RESOLUTION_METHODS = frozenset({
    "official_same_security_local_ric",
    "official_successor_local_ric",
    "official_otc_transition_local_ric",
})
_OFFICIAL_SOURCE_HOSTS = frozenset({
    "corporate.marsh.com",
    "finance.yahoo.com",
    "github.com",
    "infomemo.theocc.com",
    "investors.broadcom.com",
    "ir.echostar.com",
    "ir.lumen.com",
    "press.spglobal.com",
    "sec.gov",
    "www.bny.com",
    "www.corporate.marsh.com",
    "www.dayforce.com",
    "www.investors.dupont.com",
    "www.sec.gov",
    "www.technipfmc.com",
    "investors.thewaltdisneycompany.com",
})
_ID_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9-]*")
_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")
_DECIMAL_PATTERN = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
_RATIO_PATTERN = re.compile(
    r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|[1-9][0-9]*/[1-9][0-9]*"
)


class SecurityIdentityError(ValueError):
    """Canonical security identity evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SecurityIdentityPaths:
    """Paths for the canonical security-identity provenance bundle."""

    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))

    @classmethod
    def from_root(cls, root_dir: Path) -> "SecurityIdentityPaths":
        root = Path(root_dir)
        if root.name == "security_identity":
            return cls(root)
        return cls(root / SECURITY_IDENTITY_RELATIVE_DIR)

    @property
    def events_csv(self) -> Path:
        return self.directory / "security_events.csv"

    @property
    def legs_csv(self) -> Path:
        return self.directory / "security_event_legs.csv"

    @property
    def sources_csv(self) -> Path:
        return self.directory / "security_event_sources.csv"

    @property
    def provider_mappings_csv(self) -> Path:
        return self.directory / "provider_symbol_mappings.csv"

    @property
    def provider_row_equivalence_csv(self) -> Path:
        return self.directory / "provider_row_equivalence.csv"

    @property
    def manifest_csv(self) -> Path:
        return self.directory / "artifact_manifest.csv"

@dataclass(frozen=True, slots=True)
class SecurityIdentityBundle:
    """Validated canonical tables. Values remain their lossless CSV strings."""

    events: pd.DataFrame
    legs: pd.DataFrame
    sources: pd.DataFrame
    provider_mappings: pd.DataFrame
    provider_row_equivalence: pd.DataFrame


def _read_exact(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing canonical security identity artifact: {path}")
    frame = pd.read_csv(path, keep_default_na=False, dtype=str)
    if tuple(frame.columns) != columns:
        raise SecurityIdentityError(
            f"{path.name} must have exactly {list(columns)}; "
            f"found {list(frame.columns)}"
        )
    return frame


def _require_clean_text(frame: pd.DataFrame, *, table: str) -> None:
    for column in frame.columns:
        values = frame[column].astype(str)
        dirty = values.ne(values.str.strip()) | values.str.contains(
            r"[\r\n]", regex=True
        )
        if dirty.any():
            rows = (dirty[dirty].index + 2).tolist()
            raise SecurityIdentityError(
                f"{table}.{column} contains non-canonical text at rows {rows}"
            )


def _require_unique(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    duplicates = frame.duplicated(columns, keep=False)
    if duplicates.any():
        values = frame.loc[duplicates, columns].to_dict("records")
        raise SecurityIdentityError(f"Duplicate {label}: {values}")


def _parse_date(value: str, *, field: str, required: bool = False) -> datetime | None:
    if not value:
        if required:
            raise SecurityIdentityError(f"{field} must not be empty")
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise SecurityIdentityError(f"{field} must use YYYY-MM-DD: {value!r}") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise SecurityIdentityError(f"{field} is not canonical: {value!r}")
    return parsed


def _parse_bool(value: str, *, field: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise SecurityIdentityError(f"{field} must be True or False")


def parse_exact_decimal(
    value: str,
    *,
    field: str,
) -> Decimal | None:
    """Parse a nonnegative plain decimal without binary-float conversion."""
    if not value:
        return None
    if not _DECIMAL_PATTERN.fullmatch(value):
        raise SecurityIdentityError(
            f"{field} must be a nonnegative plain decimal: {value!r}"
        )
    try:
        return Decimal(value)
    except InvalidOperation as exc:  # pragma: no cover - guarded by regex
        raise SecurityIdentityError(f"Invalid {field}: {value!r}") from exc


def parse_exact_ratio(
    value: str,
    *,
    field: str = "Share_Ratio",
    required: bool = False,
) -> Fraction | None:
    """Parse a nonnegative decimal or exact ``numerator/denominator`` ratio."""
    if not value:
        if required:
            raise SecurityIdentityError(f"{field} must not be empty")
        return None
    if not _RATIO_PATTERN.fullmatch(value):
        raise SecurityIdentityError(
            f"{field} must be a nonnegative decimal or positive N/D ratio: {value!r}"
        )
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(value))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise SecurityIdentityError(f"Invalid {field}: {value!r}") from exc


def _validate_events(events: pd.DataFrame) -> None:
    _require_clean_text(events, table="security_events")
    _require_unique(events, ["Event_ID"], label="Event_ID")
    if events.empty:
        raise SecurityIdentityError("security_events.csv must not be empty")
    for row_number, row in enumerate(events.to_dict("records"), start=2):
        event_id = row["Event_ID"]
        if not _ID_PATTERN.fullmatch(event_id) or not event_id.startswith("EVT-"):
            raise SecurityIdentityError(f"Row {row_number}: invalid Event_ID {event_id!r}")
        _parse_date(row["Effective_Date"], field=f"Row {row_number} Effective_Date", required=True)
        if row["Event_Type"] not in ALLOWED_EVENT_TYPES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Event_Type {row['Event_Type']!r}"
            )
        if row["Continuity_Class"] not in ALLOWED_CONTINUITY_CLASSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Continuity_Class "
                f"{row['Continuity_Class']!r}"
            )
        if row["Accounting_Status"] not in ALLOWED_ACCOUNTING_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Accounting_Status "
                f"{row['Accounting_Status']!r}"
            )
        if row["Review_Status"] not in ALLOWED_REVIEW_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Review_Status {row['Review_Status']!r}"
            )
        if (
            row["Accounting_Status"] == "executable"
            and row["Review_Status"] != "approved"
        ):
            raise SecurityIdentityError(
                f"Row {row_number}: executable event must be approved"
            )
        if not row["Summary"]:
            raise SecurityIdentityError(f"Row {row_number}: Summary must not be empty")


def _validate_legs(
    events: pd.DataFrame,
    sources: pd.DataFrame,
    legs: pd.DataFrame,
) -> None:
    _require_clean_text(legs, table="security_event_legs")
    _require_unique(legs, ["Event_Leg_ID"], label="Event_Leg_ID")
    _require_unique(legs, ["Event_ID", "Leg_Sequence"], label="event leg sequence")
    event_ids = set(events["Event_ID"])
    event_by_id = events.set_index("Event_ID")
    source_index = sources.set_index("Source_ID")
    source_event = source_index["Event_ID"].to_dict()
    source_review = source_index["Review_Status"].to_dict()
    for row_number, row in enumerate(legs.to_dict("records"), start=2):
        if not _ID_PATTERN.fullmatch(row["Event_Leg_ID"]) or not row[
            "Event_Leg_ID"
        ].startswith("LEG-"):
            raise SecurityIdentityError(
                f"Row {row_number}: invalid Event_Leg_ID {row['Event_Leg_ID']!r}"
            )
        event_id = row["Event_ID"]
        if event_id not in event_ids:
            raise SecurityIdentityError(
                f"Row {row_number}: orphan Event_ID {event_id!r}"
            )
        if (
            source_event.get(row["Source_ID"]) != event_id
            or source_review.get(row["Source_ID"]) != "approved"
        ):
            raise SecurityIdentityError(
                f"Row {row_number}: Source_ID must reference an approved "
                "source for the same event"
            )
        try:
            sequence = int(row["Leg_Sequence"])
        except ValueError as exc:
            raise SecurityIdentityError(
                f"Row {row_number}: Leg_Sequence must be a positive integer"
            ) from exc
        if sequence <= 0 or str(sequence) != row["Leg_Sequence"]:
            raise SecurityIdentityError(
                f"Row {row_number}: Leg_Sequence must be a positive integer"
            )
        if not row["From_Ticker"]:
            raise SecurityIdentityError(
                f"Row {row_number}: From_Ticker must not be empty"
            )
        leg_type = row["Leg_Type"]
        if leg_type not in ALLOWED_LEG_TYPES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Leg_Type {leg_type!r}"
            )
        ratio = parse_exact_ratio(row["Share_Ratio"], field="Share_Ratio")
        executable_ratio = parse_exact_ratio(
            row["Executable_Share_Ratio"],
            field="Executable_Share_Ratio",
        )
        normalization_fields = (
            "Normalization_Denominator",
            "Normalization_Provider",
            "Normalization_Provider_Symbol",
            "Normalization_Treatment_Date",
            "Normalization_Source_ID",
        )
        normalization_values = tuple(row[field] for field in normalization_fields)
        if executable_ratio is None:
            if any(normalization_values):
                raise SecurityIdentityError(
                    f"Row {row_number}: normalization evidence requires an "
                    "Executable_Share_Ratio"
                )
        else:
            if ratio is None or ratio <= 0 or executable_ratio <= 0:
                raise SecurityIdentityError(
                    f"Row {row_number}: legal and executable share ratios must "
                    "both be positive"
                )
            if not all(normalization_values):
                raise SecurityIdentityError(
                    f"Row {row_number}: executable ratio requires complete "
                    "normalization evidence"
                )
            denominator_decimal = parse_exact_decimal(
                row["Normalization_Denominator"],
                field="Normalization_Denominator",
            )
            if denominator_decimal is None or denominator_decimal <= 0:
                raise SecurityIdentityError(
                    f"Row {row_number}: normalization denominator must be positive"
                )
            if executable_ratio != ratio / Fraction(denominator_decimal):
                raise SecurityIdentityError(
                    f"Row {row_number}: executable ratio must exactly equal "
                    "legal ratio divided by the normalization denominator"
                )
            normalization_provider = row["Normalization_Provider"]
            if normalization_provider not in _SUPPORTED_NORMALIZATION_PROVIDERS:
                raise SecurityIdentityError(
                    f"Row {row_number}: unsupported normalization provider "
                    f"{normalization_provider!r}"
                )
            treatment_date = _parse_date(
                row["Normalization_Treatment_Date"],
                field="Normalization_Treatment_Date",
                required=True,
            )
            event_date = _parse_date(
                event_by_id.loc[event_id, "Effective_Date"],
                field="Effective_Date",
                required=True,
            )
            if treatment_date < event_date:
                raise SecurityIdentityError(
                    f"Row {row_number}: normalization treatment predates the event"
                )
            normalization_source_id = row["Normalization_Source_ID"]
            if (
                source_event.get(normalization_source_id) != event_id
                or source_review.get(normalization_source_id) != "approved"
                or source_index.loc[normalization_source_id, "Source_Role"]
                != "supplemental"
                or _parse_date(
                    source_index.loc[normalization_source_id, "Document_Date"],
                    field="Document_Date",
                    required=True,
                )
                != treatment_date
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: normalization source must be approved "
                    "supplemental evidence for the same event and treatment date"
                )
            if leg_type not in {"relabel", "stock", "distribution"}:
                raise SecurityIdentityError(
                    f"Row {row_number}: normalization cannot alter cash or rights"
                )
        cash = parse_exact_decimal(row["Cash_Amount"], field="Cash_Amount")
        cvr_units = parse_exact_decimal(row["CVR_Units"], field="CVR_Units")
        cvr_base = parse_exact_decimal(
            row["CVR_Base_Value_Per_Unit"], field="CVR_Base_Value_Per_Unit"
        )
        cvr_max = parse_exact_decimal(
            row["CVR_Max_Value_Per_Unit"], field="CVR_Max_Value_Per_Unit"
        )
        retain = _parse_bool(row["Retain_Predecessor"], field="Retain_Predecessor")
        tradable = _parse_bool(row["Tradable"], field="Tradable")
        if row["Review_Status"] not in ALLOWED_REVIEW_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown leg Review_Status"
            )
        if not row["Terms_Text"]:
            raise SecurityIdentityError(f"Row {row_number}: Terms_Text must not be empty")

        if leg_type == "relabel":
            if not row["To_Ticker"] or ratio != Fraction(1) or retain:
                raise SecurityIdentityError(
                    f"Row {row_number}: relabel requires To_Ticker, ratio 1, "
                    "and Retain_Predecessor=False"
                )
        elif leg_type == "stock":
            if not row["To_Ticker"] or ratio is None or ratio <= 0 or retain:
                raise SecurityIdentityError(
                    f"Row {row_number}: stock requires a positive ratio and successor"
                )
        elif leg_type == "distribution":
            if not row["To_Ticker"] or ratio is None or ratio <= 0:
                raise SecurityIdentityError(
                    f"Row {row_number}: distribution requires a positive ratio "
                    "and successor"
                )
        elif leg_type == "cash":
            if cash is None or cash <= 0 or not row["Currency"] or row["To_Ticker"]:
                raise SecurityIdentityError(
                    f"Row {row_number}: cash requires positive Cash_Amount, Currency, "
                    "and blank To_Ticker"
                )
        elif leg_type == "cvr":
            if (
                cvr_units is None
                or cvr_units <= 0
                or cvr_base is None
                or cvr_max is None
                or cvr_base > cvr_max
                or not row["Currency"]
                or tradable
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: CVR requires positive units, ordered base/max "
                    "values, Currency, and Tradable=False"
                )
        elif leg_type == "cancellation":
            numeric = (ratio, cash, cvr_units, cvr_base, cvr_max)
            if row["To_Ticker"] or any(item is not None for item in numeric) or retain:
                raise SecurityIdentityError(
                    f"Row {row_number}: cancellation cannot create consideration"
                )

        event = event_by_id.loc[event_id]
        if event["Accounting_Status"] != "executable":
            raise SecurityIdentityError(
                f"Row {row_number}: documented_not_executable events cannot expose "
                "executable legs"
            )

    legs_by_event = set(legs["Event_ID"])
    missing = events.loc[
        events["Accounting_Status"].eq("executable")
        & ~events["Event_ID"].isin(legs_by_event),
        "Event_ID",
    ].tolist()
    if missing:
        raise SecurityIdentityError(
            f"Executable events have no structured legs: {missing}"
        )
    leg_types_by_event = {
        event_id: set(group["Leg_Type"])
        for event_id, group in legs.groupby("Event_ID", sort=False)
    }
    for event in events.loc[events["Accounting_Status"].eq("executable")].to_dict(
        "records"
    ):
        observed = leg_types_by_event[event["Event_ID"]]
        event_type = event["Event_Type"]
        allowed = {
            "identity_continuity": {"relabel"},
            "otc_transition": {"relabel"},
            "cash_settlement": {"cash", "cvr"},
            "stock_exchange": {"stock"},
            "cash_and_stock": {"cash", "stock", "cvr", "relabel"},
            "distribution": {"distribution", "relabel"},
            "cancellation": {"cancellation"},
        }[event_type]
        if not observed.issubset(allowed):
            raise SecurityIdentityError(
                f"Event {event['Event_ID']} has legs {sorted(observed)} "
                f"incompatible with {event_type}"
            )
        required = {
            "identity_continuity": {"relabel"},
            "otc_transition": {"relabel"},
            "cash_settlement": {"cash"},
            "stock_exchange": {"stock"},
            "cash_and_stock": {"cash", "stock"},
            "distribution": {"distribution"},
            "cancellation": {"cancellation"},
        }[event_type]
        if not required.issubset(observed):
            raise SecurityIdentityError(
                f"Event {event['Event_ID']} lacks required {event_type} legs"
            )
        if event_type == "distribution":
            event_legs = legs.loc[legs["Event_ID"].eq(event["Event_ID"])]
            has_relabel = event_legs["Leg_Type"].eq("relabel").any()
            distribution_retention = {
                _parse_bool(value, field="Retain_Predecessor")
                for value in event_legs.loc[
                    event_legs["Leg_Type"].eq("distribution"),
                    "Retain_Predecessor",
                ]
            }
            expected = {not has_relabel}
            if distribution_retention != expected:
                raise SecurityIdentityError(
                    f"Event {event['Event_ID']} must retain the exact predecessor "
                    "only when no relabel leg consumes it"
                )


def _validate_sources(events: pd.DataFrame, sources: pd.DataFrame) -> None:
    _require_clean_text(sources, table="security_event_sources")
    _require_unique(sources, ["Source_ID"], label="Source_ID")
    event_ids = set(events["Event_ID"])
    for row_number, row in enumerate(sources.to_dict("records"), start=2):
        if not _ID_PATTERN.fullmatch(row["Source_ID"]) or not row[
            "Source_ID"
        ].startswith("SRC-"):
            raise SecurityIdentityError(
                f"Row {row_number}: invalid Source_ID {row['Source_ID']!r}"
            )
        if row["Event_ID"] not in event_ids:
            raise SecurityIdentityError(
                f"Row {row_number}: orphan Event_ID {row['Event_ID']!r}"
            )
        if row["Source_Role"] not in ALLOWED_SOURCE_ROLES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Source_Role {row['Source_Role']!r}"
            )
        if row["Document_Date"]:
            _parse_date(row["Document_Date"], field="Document_Date")
        parsed = urlparse(row["Source_URL"])
        if parsed.scheme != "https" or parsed.hostname not in _OFFICIAL_SOURCE_HOSTS:
            raise SecurityIdentityError(
                f"Row {row_number}: source must use an allowlisted official HTTPS host"
            )
        if not row["Publisher"] or not row["Evidence_Claim"]:
            raise SecurityIdentityError(
                f"Row {row_number}: Publisher and Evidence_Claim are required"
            )
        if row["Review_Status"] not in ALLOWED_REVIEW_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown source Review_Status"
            )

    approved_primary = set(
        sources.loc[
            sources["Source_Role"].eq("primary")
            & sources["Review_Status"].eq("approved"),
            "Event_ID",
        ]
    )
    missing_sources = sorted(event_ids - approved_primary)
    if missing_sources:
        raise SecurityIdentityError(
            f"Events lack an approved primary source: {missing_sources}"
        )


def _validate_provider_mappings(
    events: pd.DataFrame,
    sources: pd.DataFrame,
    mappings: pd.DataFrame,
) -> None:
    _require_clean_text(mappings, table="provider_symbol_mappings")
    _require_unique(mappings, ["Mapping_ID"], label="Mapping_ID")
    event_ids = set(events["Event_ID"])
    source_event = sources.set_index("Source_ID")[
        "Event_ID"
    ].to_dict()
    intervals: dict[tuple[str, str, str], list[tuple[datetime, datetime | None, dict]]] = {}
    for row_number, row in enumerate(mappings.to_dict("records"), start=2):
        mapping_id = row["Mapping_ID"]
        if not _ID_PATTERN.fullmatch(mapping_id) or not mapping_id.startswith("MAP-"):
            raise SecurityIdentityError(
                f"Row {row_number}: invalid Mapping_ID {mapping_id!r}"
            )
        for field in ("Scope", "Provider", "Source_Ticker", "Asset_ID"):
            if not row[field]:
                raise SecurityIdentityError(
                    f"Row {row_number}: {field} must not be empty"
                )
        start = _parse_date(
            row["Effective_Start"], field="Effective_Start", required=True
        )
        end = _parse_date(row["Effective_End"], field="Effective_End")
        if end is not None and end <= start:
            raise SecurityIdentityError(
                f"Row {row_number}: Effective_End must be later than Effective_Start"
            )
        intervals.setdefault(
            (row["Scope"], row["Provider"], row["Source_Ticker"]), []
        ).append((start, end, row))
        if row["Resolution_Method"] not in ALLOWED_RESOLUTION_METHODS:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Resolution_Method"
            )
        event_id = row["Event_ID"]
        source_id = row["Primary_Source_ID"]
        if event_id:
            if event_id not in event_ids:
                raise SecurityIdentityError(
                    f"Row {row_number}: orphan Event_ID {event_id!r}"
                )
            if not source_id or source_event.get(source_id) != event_id:
                raise SecurityIdentityError(
                    f"Row {row_number}: Primary_Source_ID must reference the same event"
                )
        elif source_id:
            raise SecurityIdentityError(
                f"Row {row_number}: Primary_Source_ID requires Event_ID"
            )
        if row["Review_Status"] not in ALLOWED_REVIEW_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown mapping Review_Status"
            )
        _parse_bool(
            row["Simultaneous_Symbol_Conflict"],
            field="Simultaneous_Symbol_Conflict",
        )
        if row["Local_First_Date"]:
            first = _parse_date(row["Local_First_Date"], field="Local_First_Date")
            last = _parse_date(
                row["Local_Last_Date"], field="Local_Last_Date", required=True
            )
            if first > last:
                raise SecurityIdentityError(
                    f"Row {row_number}: Local_First_Date exceeds Local_Last_Date"
                )
        elif row["Local_Last_Date"]:
            raise SecurityIdentityError(
                f"Row {row_number}: Local_Last_Date requires Local_First_Date"
            )
        wikipedia_mapping = (
            row["Resolution_Method"] == "reviewed_wikipedia_mapping"
        )
        is_wikipedia_provider = row["Provider"].casefold() == "wikipedia"
        if wikipedia_mapping:
            parsed = urlparse(row["Provider_Source_URL"])
            if (
                row["Scope"].casefold() != "shared"
                or not is_wikipedia_provider
                or row["Source_Ticker"] == row["Provider_Symbol"]
                or row["Provider_Evidence_Status"]
                != "reviewed_wikipedia_historical_symbol"
                or parsed.scheme != "https"
                or parsed.hostname not in _OFFICIAL_SOURCE_HOSTS
                or not row["Provider_Evidence_Claim"]
                or row["Review_Status"] != "approved"
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: invalid reviewed Wikipedia mapping"
                )
        elif is_wikipedia_provider:
            raise SecurityIdentityError(
                f"Row {row_number}: Wikipedia provider requires "
                "reviewed_wikipedia_mapping"
            )

        missing = row["Resolution_Method"] == "reviewed_missing_price"
        if missing:
            if (
                row["Provider_Symbol"]
                or not row["Asset_ID"].startswith(UNAVAILABLE_PREFIX)
                or event_id
                or source_id
                or not row["Provider_Source_URL"]
                or not row["Provider_Evidence_Claim"]
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: invalid provider-only missing-price mapping"
                )
        elif not row["Provider_Symbol"]:
            raise SecurityIdentityError(
                f"Row {row_number}: Provider_Symbol must not be empty"
            )

        fallback_method = row["Resolution_Method"] in {
            "reviewed_yahoo_close_fallback",
            "reviewed_yahoo_event_treatment",
            "reviewed_wiki_close_fallback",
            "reviewed_local_reuters_identity",
        }
        if fallback_method:
            expected_provider = {
                "reviewed_yahoo_close_fallback": "yahoo",
                "reviewed_yahoo_event_treatment": "yahoo",
                "reviewed_wiki_close_fallback": "wiki",
                "reviewed_local_reuters_identity": "reuters",
            }[row["Resolution_Method"]]
            expected_evidence_status = (
                "local_reuters_observation"
                if row["Resolution_Method"] == "reviewed_local_reuters_identity"
                else "captured_price_observation"
            )
            if (
                row["Scope"] != "backtest"
                or row["Provider"] != expected_provider
                or not row["Provider_Source_URL"]
                or not row["Provider_Evidence_Claim"]
                or row["Provider_Evidence_Status"] != expected_evidence_status
                or row["Review_Status"] != "approved"
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: invalid reviewed backtest price mapping"
                )

        if row["Resolution_Method"] in _OFFICIAL_RESOLUTION_METHODS:
            if not event_id or not source_id:
                raise SecurityIdentityError(
                    f"Row {row_number}: official Reuters mapping requires event/source"
                )

    for key, values in intervals.items():
        ordered = sorted(values, key=lambda value: (value[0], value[1] or datetime.max))
        for index, (left_start, left_end, left) in enumerate(ordered):
            left_limit = left_end or datetime.max
            for right_start, right_end, right in ordered[index + 1 :]:
                if right_start >= left_limit:
                    break
                right_limit = right_end or datetime.max
                if left_start >= right_limit:
                    continue
                intentional = (
                    _parse_bool(
                        left["Simultaneous_Symbol_Conflict"],
                        field="Simultaneous_Symbol_Conflict",
                    )
                    and _parse_bool(
                        right["Simultaneous_Symbol_Conflict"],
                        field="Simultaneous_Symbol_Conflict",
                    )
                    and left["Asset_ID"] != right["Asset_ID"]
                )
                if not intentional:
                    raise SecurityIdentityError(
                        f"Overlapping provider mappings without approved conflict: {key}"
                    )


def _validate_provider_row_equivalence(frame: pd.DataFrame) -> None:
    _require_clean_text(frame, table="provider_row_equivalence")
    _require_unique(frame, ["Equivalence_ID"], label="Equivalence_ID")
    _require_unique(
        frame,
        ["Scope", "Provider", "Asset_ID", "Predecessor_Symbol", "Successor_Symbol"],
        label="provider row-equivalence identity",
    )
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        equivalence_id = row["Equivalence_ID"]
        if (
            not _ID_PATTERN.fullmatch(equivalence_id)
            or not equivalence_id.startswith("EQ-")
        ):
            raise SecurityIdentityError(
                f"Row {row_number}: invalid Equivalence_ID {equivalence_id!r}"
            )
        for field in (
            "Scope",
            "Provider",
            "Asset_ID",
            "Predecessor_Symbol",
            "Successor_Symbol",
        ):
            if not row[field]:
                raise SecurityIdentityError(
                    f"Row {row_number}: {field} must not be empty"
                )
        if row["Predecessor_Symbol"] == row["Successor_Symbol"]:
            raise SecurityIdentityError(
                f"Row {row_number}: equivalence symbols must differ"
            )
        effective = _parse_date(
            row["Effective_Date"], field="Effective_Date", required=True
        )
        for field in ("Absolute_Tolerance", "Relative_Tolerance"):
            parse_exact_decimal(row[field], field=field)
        for field in (
            "Predecessor_Artifact_SHA256",
            "Successor_Response_SHA256",
        ):
            if row[field] and not _HEX_PATTERN.fullmatch(row[field]):
                raise SecurityIdentityError(
                    f"Row {row_number}: {field} is not a SHA-256 digest"
                )
        if row["Review_Status"] not in ALLOWED_REVIEW_STATUSES:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown equivalence Review_Status"
            )
        result = row["Review_Result"]
        if result not in ALLOWED_EQUIVALENCE_RESULTS:
            raise SecurityIdentityError(
                f"Row {row_number}: unknown Review_Result {result!r}"
            )
        try:
            count = int(row["Common_Row_Count"] or "0")
        except ValueError as exc:
            raise SecurityIdentityError(
                f"Row {row_number}: Common_Row_Count must be nonnegative"
            ) from exc
        if count < 0 or (row["Common_Row_Count"] and str(count) != row["Common_Row_Count"]):
            raise SecurityIdentityError(
                f"Row {row_number}: Common_Row_Count must be nonnegative"
            )
        if result == "equivalent":
            try:
                fields = json.loads(row["Compared_Fields"])
            except (TypeError, json.JSONDecodeError) as exc:
                raise SecurityIdentityError(
                    f"Row {row_number}: Compared_Fields must be a JSON array"
                ) from exc
            if not isinstance(fields, list) or not fields or any(
                not isinstance(item, str) or not item for item in fields
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: Compared_Fields must be a nonempty JSON array"
                )
            common_start = _parse_date(
                row["Common_Start"], field="Common_Start", required=True
            )
            common_end = _parse_date(
                row["Common_End"], field="Common_End", required=True
            )
            if common_start > common_end or common_end >= effective:
                raise SecurityIdentityError(
                    f"Row {row_number}: equivalence overlap must precede event"
                )
            if (
                count <= 0
                or not row["Absolute_Tolerance"]
                or not row["Relative_Tolerance"]
                or not row["Predecessor_Artifact_SHA256"]
                or not row["Successor_Response_SHA256"]
                or row["Review_Status"] != "approved"
            ):
                raise SecurityIdentityError(
                    f"Row {row_number}: approved equivalence requires comparison "
                    "dates, tolerances, rows, and both hashes"
                )
        if (
            row["Predecessor_Symbol"] == "CTRA"
            and row["Successor_Symbol"] == "DVN"
            and result == "equivalent"
        ):
            raise SecurityIdentityError(
                "CTRA and DVN are different securities and cannot be historical aliases"
            )


def _validate_historical_successor_aliases(
    events: pd.DataFrame,
    legs: pd.DataFrame,
    mappings: pd.DataFrame,
    equivalence: pd.DataFrame,
) -> None:
    """Validate when a successor symbol may backfill its predecessor.

    An explicit Yahoo alias may follow a reviewed same-security 1:1 ticker
    relabel, including an exchange-to-OTC transition. Every other successor-symbol
    backfill still requires approved row-equivalence evidence. The downloaded
    snapshot separately proves which historical observations the provider returned.
    """

    event_by_id = events.set_index("Event_ID")
    approved = equivalence.loc[
        equivalence["Review_Status"].eq("approved")
        & equivalence["Review_Result"].eq("equivalent")
    ]
    approved_keys = set(zip(
        approved["Scope"],
        approved["Provider"],
        approved["Asset_ID"],
        approved["Predecessor_Symbol"],
        approved["Successor_Symbol"],
        strict=True,
    ))

    for row_number, mapping in enumerate(mappings.to_dict("records"), start=2):
        event_id = mapping["Event_ID"]
        # Row equivalence is the explicit gate for Yahoo successor-symbol
        # historical backfill. Reuters mappings instead describe the supplied,
        # immutable local series and carry their own artifact hash evidence.
        if (
            not event_id
            or mapping["Provider"].lower() != "yahoo"
            or mapping["Source_Ticker"] == mapping["Provider_Symbol"]
        ):
            continue
        event_legs = legs.loc[legs["Event_ID"].eq(event_id)]
        successor_match = event_legs.loc[
            event_legs["From_Ticker"].eq(mapping["Source_Ticker"])
            & event_legs["To_Ticker"].eq(mapping["Provider_Symbol"])
            & event_legs["Leg_Type"].isin(["relabel", "stock"])
        ]
        if successor_match.empty:
            continue
        start = _parse_date(
            mapping["Effective_Start"],
            field="Effective_Start",
            required=True,
        )
        event = event_by_id.loc[event_id]
        effective = _parse_date(
            event["Effective_Date"],
            field="Effective_Date",
            required=True,
        )
        if start >= effective:
            continue
        same_security_alias = (
            mapping["Resolution_Method"] == "reviewed_yahoo_alias"
            and mapping["Review_Status"] == "approved"
            and event["Review_Status"] == "approved"
            and event["Accounting_Status"] == "executable"
            and event["Event_Type"] in {"identity_continuity", "otc_transition"}
            and event["Continuity_Class"] == "same_security"
            and len(event_legs) == 1
            and len(successor_match) == 1
        )
        if same_security_alias:
            leg = successor_match.iloc[0]
            same_security_alias = (
                leg["Leg_Type"] == "relabel"
                and parse_exact_ratio(leg["Share_Ratio"], required=True)
                == Fraction(1)
                and not _parse_bool(
                    leg["Retain_Predecessor"], field="Retain_Predecessor"
                )
                and _parse_bool(leg["Tradable"], field="Tradable")
                and leg["Review_Status"] == "approved"
            )
        if same_security_alias:
            continue
        key = (
            mapping["Scope"],
            mapping["Provider"],
            mapping["Asset_ID"],
            mapping["Source_Ticker"],
            mapping["Provider_Symbol"],
        )
        if key not in approved_keys:
            raise SecurityIdentityError(
                f"Row {row_number}: successor symbol {mapping['Provider_Symbol']!r} "
                f"cannot backfill pre-event {mapping['Source_Ticker']!r} history "
                "without approved provider row equivalence"
            )


def validate_security_identity_bundle(
    bundle: SecurityIdentityBundle,
    *,
    require_all_approved: bool = True,
) -> SecurityIdentityBundle:
    """Validate schemas, enums, referential integrity, terms, and approvals."""
    frames = (
        (bundle.events, SECURITY_EVENTS_COLUMNS, "security_events"),
        (bundle.legs, SECURITY_EVENT_LEGS_COLUMNS, "security_event_legs"),
        (bundle.sources, SECURITY_EVENT_SOURCES_COLUMNS, "security_event_sources"),
        (
            bundle.provider_mappings,
            PROVIDER_SYMBOL_MAPPINGS_COLUMNS,
            "provider_symbol_mappings",
        ),
        (
            bundle.provider_row_equivalence,
            PROVIDER_ROW_EQUIVALENCE_COLUMNS,
            "provider_row_equivalence",
        ),
    )
    for frame, columns, label in frames:
        if tuple(frame.columns) != columns:
            raise SecurityIdentityError(
                f"{label} must have exactly {list(columns)}"
            )
    _validate_events(bundle.events)
    _validate_sources(bundle.events, bundle.sources)
    _validate_legs(bundle.events, bundle.sources, bundle.legs)
    _validate_provider_mappings(
        bundle.events, bundle.sources, bundle.provider_mappings
    )
    _validate_provider_row_equivalence(bundle.provider_row_equivalence)
    _validate_historical_successor_aliases(
        bundle.events,
        bundle.legs,
        bundle.provider_mappings,
        bundle.provider_row_equivalence,
    )
    if require_all_approved:
        unresolved: list[str] = []
        for label, frame in (
            ("event", bundle.events),
            ("leg", bundle.legs),
            ("source", bundle.sources),
            ("mapping", bundle.provider_mappings),
            ("equivalence", bundle.provider_row_equivalence),
        ):
            values = frame.loc[frame["Review_Status"].ne("approved")]
            if not values.empty:
                unresolved.append(f"{label}:{len(values)}")
        if unresolved:
            raise SecurityIdentityError(
                f"Unapproved canonical identity records remain: {unresolved}"
            )
    return bundle


def load_security_identity_bundle(
    root_dir: Path,
    *,
    validate_manifest: bool = True,
    require_all_approved: bool = True,
) -> SecurityIdentityBundle:
    """Load the canonical bundle from a project root or its exact directory."""
    paths = SecurityIdentityPaths.from_root(root_dir)
    if validate_manifest:
        validate_exact_manifest_catalog(
            paths.manifest_csv,
            scope="shared",
            dataset="security_identity",
            expected_origins={
                paths.events_csv.name: ArtifactOrigin.MANUAL,
                paths.legs_csv.name: ArtifactOrigin.MANUAL,
                paths.sources_csv.name: ArtifactOrigin.MANUAL,
                paths.provider_mappings_csv.name: ArtifactOrigin.MIGRATED,
                paths.provider_row_equivalence_csv.name: ArtifactOrigin.MANUAL,
            },
            base_dir=paths.directory,
        )
    bundle = SecurityIdentityBundle(
        events=_read_exact(paths.events_csv, SECURITY_EVENTS_COLUMNS),
        legs=_read_exact(paths.legs_csv, SECURITY_EVENT_LEGS_COLUMNS),
        sources=_read_exact(paths.sources_csv, SECURITY_EVENT_SOURCES_COLUMNS),
        provider_mappings=_read_exact(
            paths.provider_mappings_csv, PROVIDER_SYMBOL_MAPPINGS_COLUMNS
        ),
        provider_row_equivalence=_read_exact(
            paths.provider_row_equivalence_csv,
            PROVIDER_ROW_EQUIVALENCE_COLUMNS,
        ),
    )
    return validate_security_identity_bundle(
        bundle, require_all_approved=require_all_approved
    )


__all__ = [
    "ALLOWED_ACCOUNTING_STATUSES",
    "ALLOWED_CONTINUITY_CLASSES",
    "ALLOWED_EVENT_TYPES",
    "ALLOWED_LEG_TYPES",
    "PROVIDER_SYMBOL_MAPPINGS_COLUMNS",
    "PROVIDER_ROW_EQUIVALENCE_COLUMNS",
    "SECURITY_EVENTS_COLUMNS",
    "SECURITY_EVENT_LEGS_COLUMNS",
    "SECURITY_EVENT_SOURCES_COLUMNS",
    "SECURITY_IDENTITY_RELATIVE_DIR",
    "UNAVAILABLE_PREFIX",
    "SecurityIdentityBundle",
    "SecurityIdentityError",
    "SecurityIdentityPaths",
    "load_security_identity_bundle",
    "parse_exact_decimal",
    "parse_exact_ratio",
    "validate_security_identity_bundle",
]
