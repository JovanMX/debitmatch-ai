"""Structured catalog tools shared by the LangGraph squad and the MCP server.

Every financial number is produced by :func:`tools.calculator.calculate_net_impact`;
nothing here (or in any agent) derives fees or yields from free text.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple

from tools.calculator import calculate_net_impact
from tools.schemas import BankProduct

HardConstraint = Literal["zero_fee", "digital_opening", "payroll", "free_insurance", "cashback", "no_minimum_balance",
                         "minor_guardianship"]
HARD_CONSTRAINTS: Tuple[HardConstraint, ...] = (
    "zero_fee", "digital_opening", "payroll", "free_insurance", "cashback", "no_minimum_balance",
    "minor_guardianship",
)
CONSTRAINT_DESCRIPTIONS: Dict[str, str] = {
    "zero_fee": "Zero Maintenance Fee",
    "digital_opening": "Digital Opening",
    "payroll": "Payroll Compatible",
    "free_insurance": "Free Insurance",
    "cashback": "Cashback / Rewards",
    "no_minimum_balance": "No Minimum Balance",
    "minor_guardianship": "Minor Under Guardianship",
}
# Account holder: an adult (default) or a minor under guardianship (account operated by a parent / tutor).
MINOR_CONSTRAINTS = frozenset({"minor_guardianship"})
# A rate below this is a nominal yield (e.g. 0.01 %): it does not count as "pays a yield" for ranking.
MEANINGFUL_YIELD_PERCENT = 1.0


def product_net_impact(
    product: BankProduct,
    deposit: float,
    monthly_transfers: int = 2,
    preferred_channel: Optional[str] = None,
    expected_penalty_events_per_year: Optional[Mapping[str, float]] = None,
    include_breakdown: bool = True,
) -> Dict[str, Any]:
    """Run the deterministic calculator with every relevant catalog array."""
    dumped = product.model_dump(mode="json")
    return calculate_net_impact(
        deposit=float(deposit),
        yield_tiers=dumped["yield_tiers"],
        monthly_fees_detail=dumped["monthly_fees_detail"],
        estimated_monthly_transfers=int(monthly_transfers),
        spei_fee_per_transfer=product.spei_transfer_fee,
        transfer_fees_detail=dumped["transfer_fees_detail"],
        preferred_channel=preferred_channel,
        penalties_detail=dumped["penalties_detail"],
        expected_penalty_events_per_year=expected_penalty_events_per_year,
        include_breakdown=include_breakdown,
    )


def evaluate_hard_constraints(
    product: BankProduct,
    constraints: Iterable[str],
    deposit: float,
) -> Dict[str, bool]:
    """Return ``{constraint: satisfied}`` for every requested hard constraint.

    ``zero_fee`` means a $0 effective monthly maintenance fee *at the user's
    balance* (balance waivers count), computed by the calculator.
    """
    checks: Dict[str, bool] = {}
    for constraint in constraints:
        if constraint == "zero_fee":
            impact = calculate_net_impact(
                deposit=float(deposit),
                monthly_fees_detail=product.model_dump(mode="json")["monthly_fees_detail"],
                estimated_monthly_transfers=0,
                apply_isr_withholding=False,
            )
            checks[constraint] = impact["error"] is None and impact["effective_monthly_fee_mxn"] == 0.0
        elif constraint == "digital_opening":
            checks[constraint] = product.is_digital_account
        elif constraint == "payroll":
            checks[constraint] = product.is_payroll_compatible
        elif constraint == "free_insurance":
            checks[constraint] = product.has_free_insurance
        elif constraint == "cashback":
            checks[constraint] = product.has_cashback_or_perks
        elif constraint == "no_minimum_balance":
            # no opening deposit and no fee that depends on keeping an average balance
            balance_fees = any(f.cost_mxn > 0 and (f.has_minimum_balance_exemption or
                                                   "minimum" in f.service_concept.lower())
                               for f in product.monthly_fees_detail)
            checks[constraint] = product.minimum_opening_deposit == 0 and not balance_fees
        elif constraint == "minor_guardianship":
            checks[constraint] = product.for_minor_under_guardianship
        else:
            raise ValueError(f"Unknown hard constraint '{constraint}'. Allowed: {HARD_CONSTRAINTS}")
    return checks


def filter_products(
    products: Sequence[BankProduct],
    constraints: Iterable[str],
    deposit: float,
) -> Tuple[List[BankProduct], Dict[str, Dict[str, bool]]]:
    """Keep products that satisfy ALL hard constraints.

    Returns (passing products, {catalog_key: constraint checks}) - the checks
    are returned for every product so rejections stay explainable.
    """
    constraint_list = list(dict.fromkeys(constraints))
    passing: List[BankProduct] = []
    checks_by_key: Dict[str, Dict[str, bool]] = {}
    for product in products:
        checks = evaluate_hard_constraints(product, constraint_list, deposit)
        checks_by_key[product.catalog_key] = checks
        if all(checks.values()):
            passing.append(product)
    return passing, checks_by_key


def exclude_institutions(products: Sequence[BankProduct], institutions: Iterable[str]) -> List[BankProduct]:
    """Drop products of institutions the user rejected (case-insensitive, matches id or name)."""
    banned = {str(i).strip().lower() for i in institutions if str(i).strip()}
    if not banned:
        return list(products)
    return [p for p in products
            if p.institution_name.lower() not in banned and p.institution_id.lower() not in banned]


def exclude_business_accounts(products: Sequence[BankProduct], include_business: bool = False) -> List[BankProduct]:
    """Personal-advice default: drop business accounts (PFAE / persona moral) unless the user asked for one."""
    return list(products) if include_business else [p for p in products if not p.is_business_account]


def exclude_minor_only_accounts(products: Sequence[BankProduct], minor_holder: bool = False) -> List[BankProduct]:
    """Adult-advice default: drop accounts of a minor under guardianship unless the profile is for a minor."""
    return list(products) if minor_holder else [p for p in products if not p.for_minor_under_guardianship]


def profile_is_for_minor(profile: Mapping[str, Any]) -> bool:
    return bool(set(profile.get("hard_constraints") or []) & MINOR_CONSTRAINTS)


def eligible_products(products: Sequence[BankProduct], profile: Mapping[str, Any]) -> List[BankProduct]:
    """Catalog slice the squad may recommend for this profile.

    Removed: excluded banks, business accounts (unless asked for) and minor-only guardianship accounts
    (unless the account is for a minor).
    """
    personal = exclude_business_accounts(products, bool(profile.get("business_account")))
    personal = exclude_minor_only_accounts(personal, profile_is_for_minor(profile))
    return exclude_institutions(personal, profile.get("excluded_institutions") or [])


def yield_unlock(product: BankProduct, deposit: float) -> Optional[Dict[str, float]]:
    """Lowest balance ABOVE ``deposit`` at which the account pays a meaningful yield it does not pay now.

    Returns ``{"unlock_balance_mxn", "unlock_rate_percent"}`` or None. Uses the catalog yield tiers only
    (the calculator then evaluates the account at that balance).
    """
    current = 0.0
    for tier in product.yield_tiers:
        upper = tier.max_balance_mxn
        if tier.min_balance_mxn <= deposit and (upper is None or deposit <= upper):
            current = max(current, tier.annual_yield_rate)
    if current >= MEANINGFUL_YIELD_PERCENT:
        return None
    above = sorted((t for t in product.yield_tiers if t.min_balance_mxn > deposit
                    and t.annual_yield_rate >= MEANINGFUL_YIELD_PERCENT and t.annual_yield_rate > current),
                   key=lambda t: (t.min_balance_mxn, -t.annual_yield_rate))
    if not above:
        return None
    return {"unlock_balance_mxn": float(above[0].min_balance_mxn), "unlock_rate_percent": float(above[0].annual_yield_rate)}


def nearest_alternatives(
    products: Sequence[BankProduct],
    constraints: Sequence[str],
    deposit: float,
    monthly_transfers: int = 2,
    preferred_channel: Optional[str] = None,
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Criteria negotiation: products closest to satisfying ALL hard constraints.

    Ranked by number of failed constraints, then total annual fees, then net
    impact. Each entry lists which constraints must be relaxed; figures come
    from the deterministic calculator.
    """
    ranked: List[Tuple[Tuple[int, float, float, str], Dict[str, Any]]] = []
    for product in products:
        checks = evaluate_hard_constraints(product, constraints, deposit)
        failed = [c for c, ok in checks.items() if not ok]
        if not failed:
            continue
        impact = product_net_impact(product, deposit, monthly_transfers, preferred_channel)
        if impact.get("error"):
            continue
        entry = {
            "product_key": product.catalog_key,
            "institution_name": product.institution_name,
            "product_name": product.product_name,
            "relax_constraints": failed,
            "relax_labels": [CONSTRAINT_DESCRIPTIONS.get(c, c) for c in failed],
            "satisfied_constraints": [c for c, ok in checks.items() if ok],
            "total_annual_fees_mxn": impact["total_annual_fees_mxn"],
            "net_annual_impact_mxn": impact["net_annual_impact_mxn"],
            "effective_monthly_fee_mxn": impact["effective_monthly_fee_mxn"],
            "last_verified_date": product.last_verified_date.isoformat(),
        }
        key = (len(failed), impact["total_annual_fees_mxn"], -impact["net_annual_impact_mxn"], product.catalog_key)
        ranked.append((key, entry))
    ranked.sort(key=lambda item: item[0])
    return [entry for _, entry in ranked[:limit]]


def baseline_products(
    products: Sequence[BankProduct],
    constraints: Sequence[str],
    deposit: float,
    monthly_transfers: int = 2,
    preferred_channel: Optional[str] = None,
) -> List[Tuple[BankProduct, Dict[str, Any]]]:
    """Fallback Mode: zero-fee, risk-free baseline accounts from the cached catalog.

    Prefers products meeting the hard constraints; otherwise any MXN 0-fee
    account. Sorted by fees then net impact (deterministic).
    """
    scored = []
    for product in products:
        impact = product_net_impact(product, deposit, monthly_transfers, preferred_channel)
        if impact.get("error") or impact["total_annual_fees_mxn"] != 0.0:
            continue
        meets = all(evaluate_hard_constraints(product, constraints, deposit).values())
        scored.append(((not meets, -impact["net_annual_impact_mxn"], product.catalog_key), product, impact))
    scored.sort(key=lambda item: item[0])
    return [(product, impact) for _, product, impact in scored]
