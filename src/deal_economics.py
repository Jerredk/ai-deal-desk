"""Single source of truth for B2B SaaS deal-desk unit economics and approval lanes."""

from __future__ import annotations

BASE_COGS_PCT = 0.18  # 18% of list ARR → 82% margin at 0% discount
SERVICES_HOURLY_RATE = 150.0
MARGIN_FLOOR_PCT = 72.0
AUTO_APPROVE_MARGIN_PCT = 75.0
WORKING_CAPITAL_PENALTY = {
    "Net-30": 0.0,
    "Net-60": 0.015,
    "Net-90": 0.03,
}
VALID_PAYMENT_TERMS = frozenset(WORKING_CAPITAL_PENALTY.keys())
HIGH_RISK_LEGAL_FLAGS = frozenset(
    {
        "MFC_CLAUSE",
        "UNCAPPED_LIABILITY",
        "TERMINATION_CONVENIENCE_30D",
        "NON_STANDARD_AUDIT",
    }
)
VALID_SEGMENTS = frozenset({"Mid-Market", "Enterprise"})
APPROVAL_LANES = frozenset({"Auto-Approve", "VP-Sales", "CFO-And-Legal"})


def compute_list_arr(annual_contract_value: float, discount_pct: float) -> float:
    if discount_pct >= 1.0:
        raise ValueError("discount_pct must be < 1.0")
    if discount_pct < 0.0:
        raise ValueError("discount_pct must be >= 0.0")
    return annual_contract_value / (1.0 - discount_pct)


def compute_gross_margin_components(
    annual_contract_value: float,
    discount_pct: float,
    payment_terms: str,
    custom_onboarding_hours: float,
) -> dict[str, float]:
    """Return intermediate COGS components and gross_margin_pct."""
    if payment_terms not in WORKING_CAPITAL_PENALTY:
        raise ValueError(
            f"payment_terms must be one of {sorted(WORKING_CAPITAL_PENALTY)}; "
            f"got {payment_terms!r}"
        )
    list_arr = compute_list_arr(annual_contract_value, discount_pct)
    base_cogs = BASE_COGS_PCT * list_arr
    services_cogs = custom_onboarding_hours * SERVICES_HOURLY_RATE
    working_capital_penalty = (
        annual_contract_value * WORKING_CAPITAL_PENALTY[payment_terms]
    )
    margin = (
        (
            annual_contract_value
            - base_cogs
            - services_cogs
            - working_capital_penalty
        )
        / annual_contract_value
    ) * 100.0
    return {
        "list_arr": list_arr,
        "base_cogs": base_cogs,
        "services_cogs": services_cogs,
        "working_capital_penalty": working_capital_penalty,
        "gross_margin_pct": round(margin, 2),
    }


def compute_gross_margin_pct(
    annual_contract_value: float,
    discount_pct: float,
    payment_terms: str,
    custom_onboarding_hours: float,
) -> float:
    """Effective gross margin using explicit B2B SaaS unit-economics rules."""
    return compute_gross_margin_components(
        annual_contract_value,
        discount_pct,
        payment_terms,
        custom_onboarding_hours,
    )["gross_margin_pct"]


def determine_approval_lane(
    discount_pct: float,
    gross_margin_pct: float,
    payment_terms: str,
    risk_flags: list[str],
) -> str:
    """Map deal economics + legal flags to an approval lane."""
    has_high_risk = bool(set(risk_flags) & HIGH_RISK_LEGAL_FLAGS)

    if (
        discount_pct > 0.25
        or gross_margin_pct < MARGIN_FLOOR_PCT
        or payment_terms == "Net-90"
        or has_high_risk
    ):
        return "CFO-And-Legal"

    if (
        discount_pct <= 0.15
        and gross_margin_pct >= AUTO_APPROVE_MARGIN_PCT
        and payment_terms == "Net-30"
        and not has_high_risk
    ):
        return "Auto-Approve"

    if (
        0.15 < discount_pct <= 0.25
        and gross_margin_pct >= MARGIN_FLOOR_PCT
        and payment_terms in ("Net-30", "Net-60")
        and not has_high_risk
    ):
        return "VP-Sales"

    # Residual gap cases (e.g. low discount + Net-60) escalate to CFO
    return "CFO-And-Legal"


def normalize_discount_pct(discount_pct: float) -> tuple[float, str | None]:
    """Accept decimal (0.20) or percentage (20.0); return (decimal, optional note)."""
    if discount_pct > 1.0:
        normalized = discount_pct / 100.0
        if normalized >= 1.0:
            raise ValueError(
                f"discount_pct {discount_pct} normalizes to {normalized} which is >= 1.0"
            )
        note = (
            f"Interpreted discount_pct={discount_pct} as {normalized:.4f} "
            f"(percentage → decimal)."
        )
        return normalized, note
    if discount_pct < 0.0:
        raise ValueError("discount_pct must be >= 0.0")
    return discount_pct, None


def build_triggered_flags(
    gross_margin_pct: float,
    payment_terms: str,
    proposed_legal_flags: list[str] | None,
) -> list[str]:
    """Canonical risk flags triggered by economics + proposed legal redlines."""
    flags: list[str] = []
    if gross_margin_pct < MARGIN_FLOOR_PCT:
        flags.append("MARGIN_BELOW_FLOOR")
    if payment_terms == "Net-90":
        flags.append("NET_90_WORKING_CAPITAL")
    for flag in proposed_legal_flags or []:
        if flag in HIGH_RISK_LEGAL_FLAGS and flag not in flags:
            flags.append(flag)
    return flags
