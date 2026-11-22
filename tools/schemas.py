"""Pydantic v2 data contracts for the DebitMatch AI product catalog.

These models are the single source of truth for the shape of
``data/catalog.json``. Every consumer (calculator, agents, MCP server, tests)
should validate raw dictionaries through them instead of trusting raw JSON.

Rate convention: every ``*_yield_rate`` field is expressed in PERCENT
(e.g. ``10.5`` means 10.5 % annual), never as a fraction.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from datetime import date
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

logger = logging.getLogger(__name__)

NonNegativeFloat = Annotated[float, Field(ge=0.0)]
NonNegativeInt = Annotated[int, Field(ge=0)]

# Canonical strings used in ``transfer_fees_detail`` (keep in sync with the catalog).
SPEI_TRANSFER = "Interbank SPEI Transfer"
# ChromaDB document_type of the catalog's semi-structured benefit texts (atomic chunks, one per benefit).
API_BENEFIT_DOC_TYPE = "api_benefit_text"
CECOBAN_TRANSFER = "Interbank CECOBAN Transfer"
SAME_BANK_TRANSFER = "Same-Bank Transfer"

CHANNEL_BRANCH = "Branch / Teller"
CHANNEL_MOBILE_APP = "Mobile Banking App"
CHANNEL_ONLINE_WEB = "Online Banking (Web)"
CHANNEL_TELEPHONE = "Telephone Banking"
DIGITAL_CHANNELS = frozenset({CHANNEL_MOBILE_APP, CHANNEL_ONLINE_WEB})

_MONEY_TOLERANCE = 0.005


class _StrictModel(BaseModel):
    """Base model: rejects unknown keys and freezes instances."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class YieldTier(_StrictModel):
    tier_level: Annotated[int, Field(ge=1)]
    min_balance_mxn: NonNegativeFloat
    max_balance_mxn: Optional[NonNegativeFloat] = None  # None = no upper bound
    annual_yield_rate: Annotated[float, Field(ge=0.0, le=100.0)]  # percent
    payout_frequency: str
    compounding_frequency: Optional[str] = None  # capitalisation period when the source states it
    minimum_term_days: Optional[Annotated[int, Field(ge=1)]] = None  # only when > 1 day (e.g. 28-day term)
    # Filled in by DebitMatch (catalog pipeline gap-fill file) where the official table has no rate.
    estimated: bool = False

    @model_validator(mode="after")
    def _check_bounds(self) -> "YieldTier":
        if self.max_balance_mxn is not None and self.max_balance_mxn < self.min_balance_mxn:
            raise ValueError(
                f"Tier {self.tier_level}: max_balance_mxn ({self.max_balance_mxn}) "
                f"is lower than min_balance_mxn ({self.min_balance_mxn})."
            )
        return self

    def contains(self, balance: float) -> bool:
        """Inclusive range check; an open upper bound matches any larger balance."""
        upper_ok = self.max_balance_mxn is None or balance <= self.max_balance_mxn
        return self.min_balance_mxn <= balance and upper_ok


class MonthlyFee(_StrictModel):
    service_concept: str
    cost_mxn: NonNegativeFloat
    has_minimum_balance_exemption: bool
    waiver_minimum_average_balance_mxn: NonNegativeFloat = 0.0


class TransferFee(_StrictModel):
    transfer_type: str
    channel: str
    cost_per_transaction_mxn: NonNegativeFloat
    free_transactions_per_month: NonNegativeInt


class Penalty(_StrictModel):
    penalty_concept: str
    cost_mxn: NonNegativeFloat
    billing_frequency: Annotated[str, Field(min_length=1)]  # e.g. "per_event", "monthly"


class BankProduct(_StrictModel):
    """One debit / payroll account offered by a Mexican bank."""

    institution_id: Annotated[str, Field(min_length=1)]
    institution_name: Annotated[str, Field(min_length=1)]
    product_id: Annotated[str, Field(min_length=1)]
    product_name: Annotated[str, Field(min_length=1)]

    minimum_opening_deposit: NonNegativeFloat
    monthly_maintenance_fee: NonNegativeFloat
    minimum_balance_for_fee_waiver: NonNegativeFloat
    spei_transfer_fee: NonNegativeFloat
    offers_yield: bool
    annual_yield_rate: Annotated[float, Field(ge=0.0, le=100.0)]  # percent

    is_digital_account: bool
    is_payroll_compatible: bool
    has_no_maintenance_fee: bool
    has_mobile_app: bool
    has_virtual_card: bool
    has_free_insurance: bool
    has_cashback_or_perks: bool
    # Accounts for businesses (persona moral / PFAE): excluded from personal advice unless the user asks for one.
    is_business_account: bool = False
    for_minor_under_guardianship: bool = False

    qualitative_perks_summary: str
    last_verified_date: date

    yield_tiers: list[YieldTier] = Field(default_factory=list)
    monthly_fees_detail: list[MonthlyFee] = Field(default_factory=list)
    transfer_fees_detail: list[TransferFee] = Field(default_factory=list)
    penalties_detail: list[Penalty] = Field(default_factory=list)

    @property
    def catalog_key(self) -> str:
        """Globally unique key, mirrors the ``rag_docs`` folder naming."""
        return f"{self.institution_id}_{self.product_id}"


_CATALOG_ADAPTER: TypeAdapter[list[BankProduct]] = TypeAdapter(list[BankProduct])


# --------------------------------------------------------------------------- #
# Consistency audit
# --------------------------------------------------------------------------- #
class CatalogIssue(BaseModel):
    """A consistency finding between summary fields and detail arrays."""

    model_config = ConfigDict(frozen=True)

    catalog_key: str
    severity: Literal["error", "warning"]
    code: str
    message: str


class CatalogConsistencyError(ValueError):
    """Raised by ``load_catalog(strict=True)`` when hard inconsistencies exist."""

    def __init__(self, issues: list[CatalogIssue]) -> None:
        self.issues = issues
        details = "\n".join(f"  - [{i.catalog_key}] {i.code}: {i.message}" for i in issues)
        super().__init__(f"Catalog has {len(issues)} consistency error(s):\n{details}")


def _differs(a: float, b: float) -> bool:
    return abs(a - b) > _MONEY_TOLERANCE


def _fold(text: str) -> str:
    """Upper-case and strip accents ("NÓMINA" -> "NOMINA")."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).upper()


def audit_product(product: BankProduct) -> list[CatalogIssue]:
    """Cross-check the summary (top-level) fields against the detail arrays.

    The detail arrays are treated as the source of truth; summary fields are
    denormalised copies kept for fast filtering and must agree with them.

    Errors (data contradicts itself):
        FEE_SUM_MISMATCH, WAIVER_MISMATCH, NO_FEE_FLAG_MISMATCH,
        YIELD_FLAG_MISMATCH, YIELD_RATE_MISMATCH, SPEI_FEE_MISMATCH
    Warnings (suspicious but not contradictory):
        BALANCE_FEE_WITHOUT_WAIVER, PAYROLL_NAME_FLAG, NO_DIGITAL_SPEI_ENTRY
    """
    key = product.catalog_key
    issues: list[CatalogIssue] = []

    def add(severity: Literal["error", "warning"], code: str, message: str) -> None:
        issues.append(CatalogIssue(catalog_key=key, severity=severity, code=code, message=message))

    # --- Monthly fees ---
    if product.monthly_fees_detail:
        fee_sum = sum(f.cost_mxn for f in product.monthly_fees_detail)
        if _differs(product.monthly_maintenance_fee, fee_sum):
            add("error", "FEE_SUM_MISMATCH",
                f"monthly_maintenance_fee={product.monthly_maintenance_fee} but monthly_fees_detail sums {fee_sum}.")
        if product.has_no_maintenance_fee != (fee_sum <= _MONEY_TOLERANCE):
            add("error", "NO_FEE_FLAG_MISMATCH",
                f"has_no_maintenance_fee={product.has_no_maintenance_fee} but detail fees sum {fee_sum}.")

        thresholds = [
            f.waiver_minimum_average_balance_mxn
            for f in product.monthly_fees_detail
            if f.has_minimum_balance_exemption and f.waiver_minimum_average_balance_mxn > 0
        ]
        expected_waiver = max(thresholds, default=0.0)
        if _differs(product.minimum_balance_for_fee_waiver, expected_waiver):
            add("error", "WAIVER_MISMATCH",
                f"minimum_balance_for_fee_waiver={product.minimum_balance_for_fee_waiver} "
                f"but detail waiver threshold is {expected_waiver}.")

        for fee in product.monthly_fees_detail:
            folded = _fold(fee.service_concept)
            if "MINIMUM" in folded and "BALANCE" in folded and fee.cost_mxn > 0 and (
                not fee.has_minimum_balance_exemption or fee.waiver_minimum_average_balance_mxn <= 0
            ):
                add("warning", "BALANCE_FEE_WITHOUT_WAIVER",
                    f"'{fee.service_concept}' ({fee.cost_mxn} MXN) has no waiver threshold, so it is always charged.")

    # --- Yield ---
    max_tier_rate = max((t.annual_yield_rate for t in product.yield_tiers), default=0.0)
    if product.offers_yield != (max_tier_rate > 0):
        add("error", "YIELD_FLAG_MISMATCH",
            f"offers_yield={product.offers_yield} but the best tier rate is {max_tier_rate}%.")
    if product.yield_tiers and _differs(product.annual_yield_rate, max_tier_rate):
        add("error", "YIELD_RATE_MISMATCH",
            f"annual_yield_rate={product.annual_yield_rate}% but the best tier rate is {max_tier_rate}%.")

    # --- SPEI ---
    spei_entries = [t for t in product.transfer_fees_detail if t.transfer_type == SPEI_TRANSFER]
    if spei_entries:
        max_spei = max(t.cost_per_transaction_mxn for t in spei_entries)
        if _differs(product.spei_transfer_fee, max_spei):
            add("error", "SPEI_FEE_MISMATCH",
                f"spei_transfer_fee={product.spei_transfer_fee} but the highest SPEI cost listed is {max_spei}.")
        if product.has_mobile_app and not any(t.channel in DIGITAL_CHANNELS for t in spei_entries):
            add("warning", "NO_DIGITAL_SPEI_ENTRY",
                "Product has a mobile app but SPEI is only listed for non-digital channels; "
                "digital users will be priced with the branch fee.")

    # --- Flags vs. naming ---
    folded_name = _fold(product.product_name)
    if ("NOMINA" in folded_name or "PAYROLL" in folded_name) and not product.is_payroll_compatible:
        add("warning", "PAYROLL_NAME_FLAG",
            f"'{product.product_name}' looks like a payroll account but is_payroll_compatible=false.")

    return issues


def audit_catalog(products: list[BankProduct]) -> list[CatalogIssue]:
    return [issue for product in products for issue in audit_product(product)]


def load_catalog(path: str | Path, strict: bool = True) -> list[BankProduct]:
    """Load, validate and audit the catalog file.

    Args:
        path: Path to ``catalog.json``.
        strict: When True, consistency *errors* raise ``CatalogConsistencyError``.
            Warnings are always logged, never raised.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: invalid JSON, schema violation, duplicate product keys, or
            (strict mode) consistency errors.
    """
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Catalog is not valid JSON: {exc}") from exc

    products = _CATALOG_ADAPTER.validate_python(raw)  # raises pydantic.ValidationError (a ValueError)

    seen: set[str] = set()
    for product in products:
        if product.catalog_key in seen:
            raise ValueError(f"Duplicate product key in catalog: {product.catalog_key}")
        seen.add(product.catalog_key)

    issues = audit_catalog(products)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    for issue in errors:
        logger.error("Catalog error [%s] %s: %s", issue.catalog_key, issue.code, issue.message)
    for issue in warnings:
        logger.debug("Catalog warning [%s] %s: %s", issue.catalog_key, issue.code, issue.message)
    if warnings:
        logger.warning(
            "Catalog %s has %d consistency warning(s) (%s); see audit_catalog() or `ingest_docs.py --audit-catalog`.",
            catalog_path.name, len(warnings), ", ".join(sorted({i.code for i in warnings})),
        )
    if strict and errors:
        raise CatalogConsistencyError(errors)
    return products
