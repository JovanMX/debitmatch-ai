"""Deterministic annual net-impact calculator for debit / payroll accounts.

All arithmetic is done with ``decimal.Decimal`` (no binary float drift) and
results are rounded half-up to cents before being returned as ``float`` so
they are JSON-serialisable for LangGraph state and MCP payloads.

The function never raises on bad input: it returns a zeroed result with the
``error`` key populated, so an agent tool call can surface the problem to the
LLM instead of crashing the graph.

Net impact model::

    net = gross_interest - isr_withheld - (service_fees + transfer_fees + expected_penalties)
"""

from __future__ import annotations

import logging
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from tools.schemas import SPEI_TRANSFER, MonthlyFee, Penalty, TransferFee, YieldTier

logger = logging.getLogger(__name__)

_MONTHS_PER_YEAR = Decimal(12)
_HUNDRED = Decimal(100)
_CENT = Decimal("0.01")
_RATE_QUANTUM = Decimal("0.0001")

_YIELD_TIERS_ADAPTER: TypeAdapter[list[YieldTier]] = TypeAdapter(list[YieldTier])
_MONTHLY_FEES_ADAPTER: TypeAdapter[list[MonthlyFee]] = TypeAdapter(list[MonthlyFee])
_TRANSFER_FEES_ADAPTER: TypeAdapter[list[TransferFee]] = TypeAdapter(list[TransferFee])
_PENALTIES_ADAPTER: TypeAdapter[list[Penalty]] = TypeAdapter(list[Penalty])

# Occurrences per year for penalties billed on a fixed schedule.
_SCHEDULED_PENALTY_EVENTS: Dict[str, Decimal] = {
    "monthly": Decimal(12),
    "annual": Decimal(1),
    "one_time": Decimal(1),  # charged once; counted in the first year
}


class IsrParameters(BaseModel):
    """Mexican ISR withholding on interest paid by banks (personas fisicas).

    Defaults are for fiscal year 2026 and MUST be reviewed every January:
      * ``retention_rate_percent``: LIF 2026, art. 24 -> 0.90 % annual,
        applied on the CAPITAL that generates the interest (not on the interest).
      * ``uma_annual_mxn``: INEGI UMA 2026 (DOF 2026-01-09), annual value.
      * ``exempt_uma_multiple``: LISR art. 54 - no withholding for demand
        deposit / savings accounts whose average balance does not exceed
        5 annual UMAs.
      * ``cap_at_interest``: the withholding never exceeds the interest earned.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    fiscal_year: int = 2026
    retention_rate_percent: float = Field(default=0.90, ge=0.0, le=100.0)
    uma_annual_mxn: float = Field(default=42_794.64, gt=0.0)
    exempt_uma_multiple: float = Field(default=5.0, ge=0.0)
    cap_at_interest: bool = True

    @property
    def exemption_threshold_mxn(self) -> float:
        return self.uma_annual_mxn * self.exempt_uma_multiple


DEFAULT_ISR_PARAMETERS = IsrParameters()


class NetImpactResult(TypedDict):
    """Exact return contract of :func:`calculate_net_impact` (without breakdown)."""

    deposit_evaluated_mxn: float
    applicable_yield_rate_percent: float
    gross_annual_interest_mxn: float
    effective_monthly_fee_mxn: float
    fee_waived_by_balance: bool
    total_annual_fees_mxn: float
    net_annual_impact_mxn: float
    is_net_positive: bool
    error: Optional[str]


def _to_decimal(value: float | int) -> Decimal:
    # str() round-trip avoids importing binary float noise (0.1 -> 0.1000000000000000055...)
    return Decimal(str(value))


def _money(value: Decimal) -> float:
    return float(value.quantize(_CENT, rounding=ROUND_HALF_UP))


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _error_result(deposit: Any, message: str) -> Dict[str, Any]:
    logger.warning("calculate_net_impact rejected input: %s", message)
    try:
        deposit_value = float(deposit)
    except (TypeError, ValueError):
        deposit_value = 0.0
    result: NetImpactResult = {
        "deposit_evaluated_mxn": deposit_value,
        "applicable_yield_rate_percent": 0.0,
        "gross_annual_interest_mxn": 0.0,
        "effective_monthly_fee_mxn": 0.0,
        "fee_waived_by_balance": False,
        "total_annual_fees_mxn": 0.0,
        "net_annual_impact_mxn": 0.0,
        "is_net_positive": False,
        "error": message,
    }
    return dict(result)


# --------------------------------------------------------------------------- #
# Business rules
# --------------------------------------------------------------------------- #
def _select_yield_rate(deposit: Decimal, tiers: list[YieldTier]) -> Decimal:
    """Return the annual rate (percent) of the tier that contains ``deposit``.

    If tier boundaries overlap (e.g. tier 1 max == tier 2 min) the tier with
    the highest ``min_balance_mxn`` wins, i.e. the customer gets the better
    bracket they qualify for. Deposits outside every tier earn 0 %.
    """
    for tier in sorted(tiers, key=lambda t: t.min_balance_mxn, reverse=True):
        if tier.contains(float(deposit)):
            return _to_decimal(tier.annual_yield_rate)
    return Decimal(0)


def _effective_monthly_fee(deposit: Decimal, fees: list[MonthlyFee]) -> tuple[Decimal, list[str]]:
    """Sum every monthly fee, zeroing the ones waived by the deposit balance.

    Returns (effective monthly fee, list of waived concepts).
    """
    total = Decimal(0)
    waived: list[str] = []
    for fee in fees:
        waiver_threshold = _to_decimal(fee.waiver_minimum_average_balance_mxn)
        if fee.has_minimum_balance_exemption and waiver_threshold > 0 and deposit >= waiver_threshold:
            waived.append(fee.service_concept)
            continue
        total += _to_decimal(fee.cost_mxn)
    return total, waived


def _resolve_transfer_cost(
    transfers_per_month: int,
    legacy_unit_fee: Decimal,
    transfer_fees: Optional[list[TransferFee]],
    preferred_channel: Optional[str],
    transfer_type: str,
) -> Dict[str, Any]:
    """Pick the per-transfer price and free allowance for the user's channel.

    Resolution order:
      1. ``transfer_fees_detail`` not supplied -> legacy ``spei_fee_per_transfer``.
      2. Exact (transfer_type, preferred_channel) entry.
      3. Cheapest listed channel for ``transfer_type`` (a rational user would
         use it; if a preferred channel was requested this is a *fallback*).
      4. No entry for ``transfer_type`` -> legacy fee.
    """
    def legacy(source: str) -> Dict[str, Any]:
        return {"unit_cost": legacy_unit_fee, "free": 0, "channel": None, "source": source}

    if transfer_fees is None:
        resolved = legacy("spei_fee_per_transfer")
    else:
        candidates = [t for t in transfer_fees if t.transfer_type == transfer_type]
        exact = [t for t in candidates if preferred_channel and t.channel == preferred_channel]
        if exact:
            entry, source = exact[0], "exact_channel"
        elif candidates:
            entry = min(candidates, key=lambda t: (t.cost_per_transaction_mxn, -t.free_transactions_per_month))
            source = "fallback_cheapest_listed_channel" if preferred_channel else "cheapest_listed_channel"
        else:
            entry, source = None, "no_matching_entry_legacy_fee"

        resolved = (
            legacy(source)
            if entry is None
            else {
                "unit_cost": _to_decimal(entry.cost_per_transaction_mxn),
                "free": entry.free_transactions_per_month,
                "channel": entry.channel,
                "source": source,
            }
        )

    billable = max(0, transfers_per_month - resolved["free"])
    resolved["billable_per_month"] = billable
    resolved["annual_cost"] = resolved["unit_cost"] * Decimal(billable) * _MONTHS_PER_YEAR
    return resolved


def _expected_penalties(
    penalties: list[Penalty],
    expected_events: Mapping[str, float],
) -> tuple[Decimal, list[Dict[str, Any]], list[str]]:
    """Expected annual penalty cost.

    Scheduled frequencies (monthly / annual / one_time) are always applied.
    ``per_event`` (and unknown frequencies) use ``expected_events[concept]``
    (events per year, default 0 -> the user is assumed to avoid them).
    """
    total = Decimal(0)
    considered: list[Dict[str, Any]] = []
    notes: list[str] = []
    for penalty in penalties:
        frequency = penalty.billing_frequency.lower()
        if frequency in _SCHEDULED_PENALTY_EVENTS:
            events = _SCHEDULED_PENALTY_EVENTS[frequency]
        else:
            if frequency != "per_event":
                notes.append(f"Unknown billing_frequency '{penalty.billing_frequency}' treated as per_event.")
            events = _to_decimal(expected_events.get(penalty.penalty_concept, 0.0))
        cost = _to_decimal(penalty.cost_mxn) * events
        total += cost
        considered.append(
            {"penalty_concept": penalty.penalty_concept, "events_per_year": float(events), "annual_cost_mxn": _money(cost)}
        )
    return total, considered, notes


def _isr_withholding(deposit: Decimal, gross_interest: Decimal, params: IsrParameters) -> tuple[Decimal, bool]:
    """Return (ISR withheld, is_exempt)."""
    if deposit <= _to_decimal(params.exemption_threshold_mxn):
        return Decimal(0), True
    withheld = deposit * _to_decimal(params.retention_rate_percent) / _HUNDRED
    if params.cap_at_interest:
        withheld = min(withheld, gross_interest)
    return withheld, False


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def calculate_net_impact(
    deposit: float,
    yield_tiers: Optional[List[Dict[str, Any]]] = None,
    monthly_fees_detail: Optional[List[Dict[str, Any]]] = None,
    estimated_monthly_transfers: int = 2,
    spei_fee_per_transfer: float = 0.0,
    *,
    transfer_fees_detail: Optional[List[Dict[str, Any]]] = None,
    preferred_channel: Optional[str] = None,
    transfer_type: str = SPEI_TRANSFER,
    penalties_detail: Optional[List[Dict[str, Any]]] = None,
    expected_penalty_events_per_year: Optional[Mapping[str, float]] = None,
    apply_isr_withholding: bool = True,
    isr_parameters: IsrParameters = DEFAULT_ISR_PARAMETERS,
    include_breakdown: bool = False,
) -> Dict[str, Any]:
    """Compute the exact annual financial impact (MXN) of holding ``deposit``.

    Args:
        deposit: Average balance the customer keeps in the account (MXN, >= 0).
        yield_tiers: ``yield_tiers`` array from the catalog (percent rates).
        monthly_fees_detail: ``monthly_fees_detail`` array from the catalog.
        estimated_monthly_transfers: Outgoing transfers per month (>= 0).
        spei_fee_per_transfer: Legacy flat price per transfer; used only when
            ``transfer_fees_detail`` is not supplied or has no matching entry.

    Keyword-only extensions:
        transfer_fees_detail: Catalog array; enables channel-aware pricing and
            ``free_transactions_per_month``.
        preferred_channel: e.g. ``"Mobile Banking App"`` (see ``tools.schemas``).
        transfer_type: Transfer type to price (default SPEI).
        penalties_detail: Catalog array of penalties.
        expected_penalty_events_per_year: ``{penalty_concept: events/year}``
            for ``per_event`` penalties (missing concepts -> 0).
        apply_isr_withholding: Deduct the ISR withheld on interest (default True).
        isr_parameters: Fiscal parameters (defaults: FY2026).
        include_breakdown: Add a ``breakdown`` key with the itemised calculation.

    Returns:
        A dict matching :class:`NetImpactResult` (plus ``breakdown`` if asked).
        ``error`` is ``None`` on success.
    """
    # --- Scalar input validation (bool is an int subclass: reject explicitly) ---
    if not _is_number(deposit):
        return _error_result(deposit, f"deposit must be a number, got {type(deposit).__name__}.")
    if isinstance(estimated_monthly_transfers, bool) or not isinstance(estimated_monthly_transfers, int):
        return _error_result(deposit, "estimated_monthly_transfers must be an integer.")
    if not _is_number(spei_fee_per_transfer):
        return _error_result(deposit, "spei_fee_per_transfer must be a number.")
    if not isinstance(isr_parameters, IsrParameters):
        return _error_result(deposit, "isr_parameters must be an IsrParameters instance.")

    try:
        deposit_d = _to_decimal(deposit)
        spei_fee_d = _to_decimal(spei_fee_per_transfer)
    except InvalidOperation:
        return _error_result(deposit, "deposit or spei_fee_per_transfer is not a finite number.")

    if not deposit_d.is_finite() or not spei_fee_d.is_finite():
        return _error_result(deposit, "deposit and spei_fee_per_transfer must be finite.")
    if deposit_d < 0:
        return _error_result(deposit, "deposit cannot be negative.")
    if spei_fee_d < 0:
        return _error_result(deposit, "spei_fee_per_transfer cannot be negative.")
    if estimated_monthly_transfers < 0:
        return _error_result(deposit, "estimated_monthly_transfers cannot be negative.")

    events_map: Mapping[str, float] = expected_penalty_events_per_year or {}
    for concept, events in events_map.items():
        if not _is_number(events) or events < 0 or events != events:  # NaN check
            return _error_result(deposit, f"expected events for '{concept}' must be a non-negative number.")

    # --- Nested array validation through the shared Pydantic contracts ---
    try:
        tiers = _YIELD_TIERS_ADAPTER.validate_python(yield_tiers or [])
        fees = _MONTHLY_FEES_ADAPTER.validate_python(monthly_fees_detail or [])
        transfer_fees = (
            None if transfer_fees_detail is None else _TRANSFER_FEES_ADAPTER.validate_python(transfer_fees_detail)
        )
        penalties = _PENALTIES_ADAPTER.validate_python(penalties_detail or [])
    except ValidationError as exc:
        return _error_result(deposit, f"Invalid catalog detail arrays: {exc.errors(include_url=False)}")

    # --- Income ---
    rate_percent = _select_yield_rate(deposit_d, tiers)
    gross_interest = deposit_d * rate_percent / _HUNDRED
    if apply_isr_withholding:
        isr_withheld, isr_exempt = _isr_withholding(deposit_d, gross_interest, isr_parameters)
    else:
        isr_withheld, isr_exempt = Decimal(0), False

    # --- Costs ---
    monthly_fee, waived_concepts = _effective_monthly_fee(deposit_d, fees)
    annual_service_fees = monthly_fee * _MONTHS_PER_YEAR
    transfer = _resolve_transfer_cost(
        estimated_monthly_transfers, spei_fee_d, transfer_fees, preferred_channel, transfer_type
    )
    annual_penalties, penalties_considered, notes = _expected_penalties(penalties, events_map)
    total_annual_fees = annual_service_fees + transfer["annual_cost"] + annual_penalties

    net_before_tax = gross_interest - total_annual_fees
    net_rounded = _money(net_before_tax - isr_withheld)

    result: Dict[str, Any] = dict(
        NetImpactResult(
            deposit_evaluated_mxn=_money(deposit_d),
            applicable_yield_rate_percent=float(rate_percent.quantize(_RATE_QUANTUM, rounding=ROUND_HALF_UP)),
            gross_annual_interest_mxn=_money(gross_interest),
            effective_monthly_fee_mxn=_money(monthly_fee),
            fee_waived_by_balance=bool(waived_concepts),
            total_annual_fees_mxn=_money(total_annual_fees),
            net_annual_impact_mxn=net_rounded,
            is_net_positive=net_rounded > 0,
            error=None,
        )
    )

    if include_breakdown:
        if transfer["source"] == "fallback_cheapest_listed_channel":
            notes.append(
                f"'{transfer_type}' is not listed for channel '{preferred_channel}'; "
                f"priced with '{transfer['channel']}'."
            )
        result["breakdown"] = {
            "annual_service_fees_mxn": _money(annual_service_fees),
            "waived_fee_concepts": waived_concepts,
            "transfer_type": transfer_type,
            "transfer_channel_used": transfer["channel"],
            "transfer_fee_source": transfer["source"],
            "transfer_unit_cost_mxn": _money(transfer["unit_cost"]),
            "billable_transfers_per_month": transfer["billable_per_month"],
            "annual_transfer_fees_mxn": _money(transfer["annual_cost"]),
            "annual_expected_penalties_mxn": _money(annual_penalties),
            "penalties_considered": penalties_considered,
            "net_before_tax_mxn": _money(net_before_tax),
            "isr_applied": apply_isr_withholding,
            "isr_exempt": isr_exempt,
            "isr_withheld_mxn": _money(isr_withheld),
            "isr_fiscal_year": isr_parameters.fiscal_year,
            "notes": notes,
        }
    return result
