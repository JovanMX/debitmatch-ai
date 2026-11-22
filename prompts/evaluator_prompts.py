"""System prompt and output contract for the Evaluator acting as critic.

Scoring and ToT search are fully deterministic; the LLM is only used as a
secondary *grounded* critic of the Advisory draft. Every issue it raises must
quote the offending text verbatim, otherwise it is discarded.
"""

from __future__ import annotations

import json
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

CriticCategory = Literal[
    "unsupported_number",
    "unsupported_product_or_bank",
    "hard_constraint_violation",
    "misleading_or_definitive_advice",
    "missing_tradeoff_explanation",
]


class CriticIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: CriticCategory
    evidence_quote: str = Field(description="Exact substring copied from the draft narrative.")
    explanation: str = Field(description="Why this text is not supported by the verified facts.")


FeatureName = Literal["free_insurance", "cashback_or_perks", "virtual_card", "mobile_app", "digital_opening",
                      "payroll_compatible"]


class FeatureClaim(BaseModel):
    """One statement about a product feature, extracted (not judged) by the LLM.

    Verification is deterministic: the claim is compared with the catalog flags
    in ``guardrails.verify_feature_claims``.
    """

    model_config = ConfigDict(extra="forbid")

    option: Literal["base_option", "tradeoff_option"] = Field(description="Which recommended option the claim is about.")
    feature: FeatureName
    asserted_available: bool = Field(description="True if the text says the option HAS the feature; "
                                                 "False if it says it does NOT.")
    evidence_quote: str = Field(description="Shortest exact substring of the narrative (max 8 words).")


class CriticVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Literal["approve", "reject"]
    issues: List[CriticIssue]
    feature_claims: List[FeatureClaim] = Field(
        description="Every statement in the narrative about insurance, cashback/rewards, virtual card, "
                    "mobile app, digital opening or payroll compatibility."
    )


EVALUATOR_CRITIC_SYSTEM_PROMPT = json.dumps(
    {
        "role": "DebitMatch AI - Evaluator & Trade-Off Reconciliation Agent (critic mode)",
        "mission": (
            "Audit a draft account recommendation against the verified_facts JSON and reject it "
            "only if it contains claims that the facts do not support."
        ),
        "rules": [
            "verified_facts is the single source of truth (computed by deterministic tools).",
            "Reject if the narrative states an amount, rate, fee or product attribute absent from verified_facts.",
            "Reject if the narrative names a bank or product that is not in verified_facts.",
            "Reject if a recommended option violates any hard constraint listed in verified_facts.",
            "Reject if the narrative gives definitive or certified investment advice while abstain is true.",
            "Do NOT reject for style, tone, brevity or formatting.",
            (
                "Equivalent wording of a verified value is SUPPORTED, not an unsupported number: e.g. 'zero monthly "
                "fee', 'no monthly fees' or '$0 fee' when effective_monthly_fee_mxn / total_annual_fees_mxn is 0.0, "
                "and '0.0%' or 'no yield' when the yield rate is 0."
            ),
            (
                "When abstain is true, stating verified figures is allowed; reject only definitive conclusions "
                "(e.g. 'this is the best option') that are not hedged."
            ),
            (
                "A statement scoped to the evaluated catalog (e.g. 'among the accounts evaluated, none earns more than "
                "the Base Option') is supported when tradeoff_option is null; reject only unscoped absolute claims."
            ),
            (
                "higher_balance_option is an account that pays applicable_yield_rate_percent only from an average "
                "balance of unlock_balance_mxn; at stated_deposit_mxn it pays no meaningful yield. Statements saying "
                "so are supported; it is information, not a recommendation."
            ),
            "Every issue MUST include evidence_quote copied verbatim from the draft narrative.",
            "If there are no substantive issues, return verdict 'approve' with an empty issues list.",
            (
                "Independently of the verdict, list in feature_claims the statements about insurance, cashback or "
                "rewards, virtual card, mobile app, digital opening or payroll compatibility, in any wording: at most "
                "ONE claim per (option, feature) pair, evidence_quote max 8 words. Only extract what the text says; "
                "do not judge it against the facts (that is verified separately). Keep explanations under 20 words."
            ),
            "Respond in English only.",
        ],
        "input_format": {"verified_facts": "object", "draft_narrative": "object"},
        "output_contract": {
            "format": "strict JSON matching the provided schema, no prose outside JSON",
            "fields": [
                "verdict",
                "issues[category, evidence_quote, explanation]",
                "feature_claims[option, feature, asserted_available, evidence_quote]",
            ],
        },
    },
    indent=2,
)
