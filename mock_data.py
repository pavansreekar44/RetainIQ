"""
mock_data.py — Data-Contract Layer for the AI Customer Churn Retention Agent
=============================================================================

This module is the **single source of truth** for every data shape exchanged
between the ML pipeline, the LLM-based Agent pipeline, and the Streamlit UI.

All functions return plain Python structures (dicts / lists of dicts) so they
can be swapped out for real implementations later with zero UI changes.

Schema is restricted to **7 universal subscription features** (plus
``customer_id``) that generalise across B2B SaaS and Telecom businesses,
preventing ML context bloat and keeping the LLM agent focused.

Dataset reference: IBM Telco Customer Churn
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

from __future__ import annotations

from typing import Any, TypedDict


# ---------------------------------------------------------------------------
# Strict schema — exactly 8 keys (customer_id + 7 subscription features)
# ---------------------------------------------------------------------------

class CustomerRecord(TypedDict):
    """Typed contract for a single customer row.

    Contains exactly 8 keys: ``customer_id`` plus the 7 universal
    subscription features that generalise across B2B SaaS and Telecom.
    Swapping mock data for real ``pandas.DataFrame.to_dict("records")``
    output requires zero transformation.
    """

    customer_id: str
    Contract: str                # "Month-to-month" | "One year" | "Two year"
    tenure: int                  # 1–72 months
    MonthlyCharges: float        # current monthly bill (USD)
    TotalCharges: float          # lifetime spend (USD)
    PaymentMethod: str           # "Electronic check" | "Mailed check" | "Bank transfer" | "Credit card"
    TechSupport: str             # "Yes" | "No"
    PaperlessBilling: str        # "Yes" | "No"


# ---------------------------------------------------------------------------
# Internal constants — 5 mock customer records
# ---------------------------------------------------------------------------

_CUSTOMERS: list[CustomerRecord] = [
    {
        "customer_id": "CUST-1024",
        "Contract": "Month-to-month",
        "tenure": 3,
        "MonthlyCharges": 89.50,
        "TotalCharges": 268.50,
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
    },
    {
        "customer_id": "CUST-1037",
        "Contract": "Two year",
        "tenure": 48,
        "MonthlyCharges": 42.30,
        "TotalCharges": 2030.40,
        "PaymentMethod": "Bank transfer",
        "TechSupport": "Yes",
        "PaperlessBilling": "No",
    },
    {
        "customer_id": "CUST-2048",
        "Contract": "Month-to-month",
        "tenure": 7,
        "MonthlyCharges": 95.75,
        "TotalCharges": 670.25,
        "PaymentMethod": "Credit card",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
    },
    {
        "customer_id": "CUST-3001",
        "Contract": "One year",
        "tenure": 60,
        "MonthlyCharges": 55.00,
        "TotalCharges": 3300.00,
        "PaymentMethod": "Mailed check",
        "TechSupport": "Yes",
        "PaperlessBilling": "No",
    },
    {
        "customer_id": "CUST-4096",
        "Contract": "Month-to-month",
        "tenure": 1,
        "MonthlyCharges": 105.20,
        "TotalCharges": 105.20,
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "PaperlessBilling": "Yes",
    },
]

# Quick lookup by ID
_CUSTOMER_INDEX: dict[str, CustomerRecord] = {
    c["customer_id"]: c for c in _CUSTOMERS
}

# The 7 allowed feature keys (everything except customer_id).
# Used as a runtime guard — SHAP feature_name MUST be one of these.
ALLOWED_FEATURE_KEYS: frozenset[str] = frozenset(
    CustomerRecord.__annotations__.keys() - {"customer_id"}
)


# ---------------------------------------------------------------------------
# SHAP reason templates — keyed by exact dataset column name
# Each entry carries a base impact value and human-readable description.
# ---------------------------------------------------------------------------

_SHAP_LIBRARY: dict[str, dict[str, Any]] = {
    "Contract": {
        "impact_value": 0.38,
        "description": (
            "Month-to-month contract increases churn risk significantly."
        ),
    },
    "TechSupport": {
        "impact_value": 0.27,
        "description": (
            "Lack of tech support correlates with higher churn probability."
        ),
    },
    "MonthlyCharges": {
        "impact_value": 0.22,
        "description": (
            "High monthly charges relative to peers increase churn likelihood."
        ),
    },
    "tenure": {
        "impact_value": 0.18,
        "description": (
            "Short tenure indicates the customer hasn't yet built loyalty."
        ),
    },
    "PaymentMethod": {
        "impact_value": 0.13,
        "description": (
            "Electronic check users churn at higher rates than auto-pay users."
        ),
    },
    "PaperlessBilling": {
        "impact_value": 0.09,
        "description": (
            "Paperless billing customers tend to disengage and churn faster."
        ),
    },
    "TotalCharges": {
        "impact_value": 0.05,
        "description": (
            "Low lifetime spend signals limited engagement and higher churn risk."
        ),
    },
}

# Compile-time sanity check: every SHAP key must be an allowed feature key
assert set(_SHAP_LIBRARY.keys()) <= ALLOWED_FEATURE_KEYS, (
    f"SHAP keys {set(_SHAP_LIBRARY.keys()) - ALLOWED_FEATURE_KEYS} "
    f"are not in the allowed feature set."
)


# ---------------------------------------------------------------------------
# 1. Customer list
# ---------------------------------------------------------------------------

def get_mock_customer_list() -> list[CustomerRecord]:
    """Return the full mock customer database.

    Each dictionary in the returned list contains **exactly** the 8
    contract-mandated keys defined in :class:`CustomerRecord`:

    * ``customer_id`` — unique string identifier
    * ``Contract`` — one of *Month-to-month*, *One year*, *Two year*
    * ``tenure`` — months the customer has been active (1–72)
    * ``MonthlyCharges`` — current monthly bill (USD)
    * ``TotalCharges`` — lifetime spend (USD)
    * ``PaymentMethod`` — payment channel
    * ``TechSupport`` — *Yes* or *No*
    * ``PaperlessBilling`` — *Yes* or *No*

    Returns
    -------
    list[CustomerRecord]
        Five realistic customer records.
    """
    return [dict(c) for c in _CUSTOMERS]  # type: ignore[misc]  # shallow copies


# ---------------------------------------------------------------------------
# 2. Churn prediction
# ---------------------------------------------------------------------------

def get_mock_churn_prediction(customer_id: str) -> dict[str, str | float | bool]:
    """Simulate an XGBoost binary-classification prediction for *customer_id*.

    Deterministic rule (so every team member gets reproducible results):
    * If the **last digit** of *customer_id* is **even** → high risk
      (``churn_risk_score`` > 0.75, ``is_high_risk`` = True).
    * Otherwise → low risk (score < 0.40).

    Parameters
    ----------
    customer_id : str
        The unique customer identifier (e.g. ``"CUST-1024"``).

    Returns
    -------
    dict[str, str | float | bool]
        Keys: ``customer_id``, ``churn_risk_score``, ``is_high_risk``.
    """
    # Grab last character that is a digit
    last_digit: int = int("".join(filter(str.isdigit, customer_id))[-1])
    is_even: bool = last_digit % 2 == 0

    risk_score: float = (
        round(0.82 + (last_digit % 5) * 0.03, 2)
        if is_even
        else round(0.18 + (last_digit % 4) * 0.05, 2)
    )

    return {
        "customer_id": customer_id,
        "churn_risk_score": risk_score,
        "is_high_risk": is_even,
    }


# ---------------------------------------------------------------------------
# 3. SHAP reasons
# ---------------------------------------------------------------------------

def get_mock_shap_reasons(customer_id: str) -> list[dict[str, str | float]]:
    """Return the top-3 SHAP feature-importance reasons for *customer_id*.

    Every ``feature_name`` in the returned dicts is guaranteed to be one of
    the 7 allowed feature keys defined in :class:`CustomerRecord`.

    The features are ranked contextually based on the customer's profile
    so the story is coherent end-to-end:

    * A month-to-month customer will see **Contract** boosted.
    * A customer with no tech support will see **TechSupport** boosted.
    * A high-charge customer will see **MonthlyCharges** boosted.

    If the customer ID is unknown the function still returns sensible
    defaults (top-3 by base impact) so the UI never breaks.

    Parameters
    ----------
    customer_id : str
        The unique customer identifier.

    Returns
    -------
    list[dict[str, str | float]]
        Up to 3 dicts with keys: ``feature_name``, ``impact_value``,
        ``description``.
    """
    customer: CustomerRecord | None = _CUSTOMER_INDEX.get(customer_id)

    # Score each SHAP feature for relevance to this customer
    ranked: list[tuple[str, dict[str, Any]]] = []

    if customer is None:
        # Unknown ID — return generic top-3 features by impact
        ranked = sorted(
            _SHAP_LIBRARY.items(),
            key=lambda kv: kv[1]["impact_value"],
            reverse=True,
        )
    else:
        for feature, info in _SHAP_LIBRARY.items():
            boost: float = 0.0

            if feature == "Contract" and customer["Contract"] == "Month-to-month":
                boost = 0.20
            elif feature == "TechSupport" and customer["TechSupport"] == "No":
                boost = 0.15
            elif feature == "MonthlyCharges" and customer["MonthlyCharges"] > 80.0:
                boost = 0.12
            elif feature == "tenure" and customer["tenure"] < 12:
                boost = 0.10
            elif feature == "PaymentMethod" and customer["PaymentMethod"] == "Electronic check":
                boost = 0.08
            elif feature == "PaperlessBilling" and customer["PaperlessBilling"] == "Yes":
                boost = 0.06
            elif feature == "TotalCharges" and customer["TotalCharges"] < 500.0:
                boost = 0.07

            ranked.append(
                (feature, {**info, "impact_value": round(info["impact_value"] + boost, 2)})
            )
        ranked.sort(key=lambda kv: kv[1]["impact_value"], reverse=True)

    top_3 = ranked[:3]
    return [
        {
            "feature_name": name,
            "impact_value": data["impact_value"],
            "description": data["description"],
        }
        for name, data in top_3
    ]


# ---------------------------------------------------------------------------
# 4. Agent intervention
# ---------------------------------------------------------------------------

def get_mock_agent_intervention(
    customer_id: str,
    risk_score: float,
    shap_reasons: list[dict[str, str | float]],
) -> dict[str, str | dict[str, str]]:
    """Simulate an LLM-generated retention strategy for *customer_id*.

    The generated email and recommended action are **coherent** with the
    supplied SHAP reasons.  For example, if *MonthlyCharges* is a top
    driver the email will offer a discount; if *TechSupport* is a driver
    it will offer complimentary support.

    Parameters
    ----------
    customer_id : str
        The unique customer identifier.
    risk_score : float
        The churn probability returned by ``get_mock_churn_prediction``.
    shap_reasons : list[dict[str, str | float]]
        The top SHAP reasons returned by ``get_mock_shap_reasons``.

    Returns
    -------
    dict[str, str | dict[str, str]]
        Keys:
        * ``generated_email_draft`` — 3-sentence personalised retention email.
        * ``recommended_action``    — internal note for the retention team.
        * ``api_payload``           — mock JSON body for the billing system.
    """
    # ---- Derive the dominant reason to tailor the response ----
    top_feature: str = shap_reasons[0]["feature_name"] if shap_reasons else "General"

    # Personalised content mapped to the dominant SHAP feature key
    _INTERVENTION_MAP: dict[str, dict[str, Any]] = {
        "MonthlyCharges": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "We truly value your loyalty and noticed your current plan may not be "
                "the most cost-effective option for you. We'd love to offer you an "
                "exclusive 20% discount on your monthly bill for the next 6 months. "
                "Please reply to this email or call us at 1-800-RETAIN to activate "
                "your savings today!"
            ),
            "action": (
                f"Apply 20% loyalty discount to {customer_id}. "
                "Escalate to Retention Tier-2 if customer declines initial offer."
            ),
            "payload": {
                "discount_code": "SAVE20",
                "discount_percent": "20",
                "duration_months": "6",
                "apply_to": customer_id,
            },
        },
        "Contract": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "We see you're currently on a flexible month-to-month plan. "
                "We'd like to reward your continued trust with a special annual "
                "contract offer that locks in a lower rate and includes priority "
                "support. Upgrade today and save up to 15% — simply reply or call "
                "1-800-RETAIN to get started!"
            ),
            "action": (
                f"Offer annual contract migration to {customer_id} with 15% rate "
                "reduction. Flag for follow-up call within 48 hours."
            ),
            "payload": {
                "discount_code": "ANNUAL15",
                "discount_percent": "15",
                "new_contract_type": "One year",
                "apply_to": customer_id,
            },
        },
        "TechSupport": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "Your experience matters to us, and we'd like to make sure you "
                "always have help when you need it. We're pleased to offer you "
                "3 months of complimentary Premium Tech Support — including 24/7 "
                "priority assistance. Activate your free upgrade by replying to "
                "this email or calling 1-800-RETAIN!"
            ),
            "action": (
                f"Provision complimentary Premium Tech Support for {customer_id} "
                "for 90 days. Schedule a satisfaction check-in at Day 30."
            ),
            "payload": {
                "discount_code": "TECHFREE90",
                "addon": "Premium Tech Support",
                "duration_months": "3",
                "apply_to": customer_id,
            },
        },
        "tenure": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "Welcome aboard — we're excited to have you! As a new member, "
                "we'd love to set you up for success with a complimentary onboarding "
                "session and a 10% loyalty bonus on your next 3 bills. Reply or "
                "call 1-800-RETAIN and we'll get you started right away!"
            ),
            "action": (
                f"Trigger new-customer nurture sequence for {customer_id}. "
                "Assign dedicated onboarding specialist."
            ),
            "payload": {
                "discount_code": "WELCOME10",
                "discount_percent": "10",
                "duration_months": "3",
                "apply_to": customer_id,
            },
        },
        "PaymentMethod": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "We noticed you're paying via electronic check — switching to "
                "autopay could save you hassle and earn you a $5/month credit. "
                "It's quick to set up and ensures you never miss a payment. "
                "Reply or call 1-800-RETAIN and we'll handle the switch for you!"
            ),
            "action": (
                f"Offer autopay migration incentive to {customer_id}. "
                "Apply $5/month credit for 12 months upon enrollment."
            ),
            "payload": {
                "discount_code": "AUTOPAY5",
                "credit_per_month": "5.00",
                "duration_months": "12",
                "apply_to": customer_id,
            },
        },

        "PaperlessBilling": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "We know managing digital bills can sometimes feel impersonal. "
                "We'd like to offer you a dedicated account manager and a $3/month "
                "billing credit as a thank-you for going paperless. Reply or call "
                "1-800-RETAIN and we'll set it up for you!"
            ),
            "action": (
                f"Assign account manager to {customer_id}. "
                "Apply $3/month paperless loyalty credit for 6 months."
            ),
            "payload": {
                "discount_code": "PAPER3",
                "credit_per_month": "3.00",
                "duration_months": "6",
                "apply_to": customer_id,
            },
        },
        "TotalCharges": {
            "email": (
                f"Dear Customer {customer_id},\n\n"
                "We noticed you're still getting started with us and want to make "
                "sure you see the full value of your plan. We're offering a one-time "
                "$25 account credit plus a free service review session. Reply or call "
                "1-800-RETAIN and let's make sure your plan fits your needs!"
            ),
            "action": (
                f"Apply $25 one-time credit to {customer_id}. "
                "Schedule plan optimisation call within 72 hours."
            ),
            "payload": {
                "discount_code": "BOOST25",
                "one_time_credit": "25.00",
                "apply_to": customer_id,
            },
        },
    }

    # Fall back to a generic high-risk intervention
    fallback: dict[str, Any] = {
        "email": (
            f"Dear Customer {customer_id},\n\n"
            "We value you as a customer and want to make sure you're getting "
            "the best possible experience. We have a special offer just for "
            "you — please reply or call 1-800-RETAIN so we can discuss how "
            "to improve your plan today!"
        ),
        "action": (
            f"Generic retention outreach for {customer_id}. "
            "Schedule manager callback within 24 hours."
        ),
        "payload": {
            "discount_code": "LOYALTY10",
            "discount_percent": "10",
            "apply_to": customer_id,
        },
    }

    intervention: dict[str, Any] = _INTERVENTION_MAP.get(top_feature, fallback)

    return {
        "generated_email_draft": intervention["email"],
        "recommended_action": intervention["action"],
        "api_payload": intervention["payload"],
    }


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 72)
    print("MOCK DATA — Smoke Test  (strict 7-feature schema)")
    print("=" * 72)

    # Verify schema compliance
    expected_keys = set(CustomerRecord.__annotations__.keys())
    for c in _CUSTOMERS:
        actual = set(c.keys())
        assert actual == expected_keys, (
            f"{c['customer_id']}: key mismatch — "
            f"extra={actual - expected_keys}, missing={expected_keys - actual}"
        )
    print("\n✅  All 5 customers match the CustomerRecord schema.")

    customers = get_mock_customer_list()
    print(f"📋  {len(customers)} customers loaded.\n")

    for cust in customers:
        cid: str = cust["customer_id"]
        prediction = get_mock_churn_prediction(cid)
        reasons = get_mock_shap_reasons(cid)
        intervention = get_mock_agent_intervention(
            cid, prediction["churn_risk_score"], reasons
        )

        # Verify SHAP feature names are in the allowed set
        for r in reasons:
            assert r["feature_name"] in ALLOWED_FEATURE_KEYS, (
                f"SHAP feature '{r['feature_name']}' not in allowed keys!"
            )

        risk_label = "🔴 HIGH" if prediction["is_high_risk"] else "🟢 LOW"
        print(
            f"--- {cid} | tenure: {cust['tenure']}mo | "
            f"${cust['MonthlyCharges']}/mo | ${cust['TotalCharges']} total | "
            f"{cust['Contract']} ---"
        )
        print(f"    Risk: {prediction['churn_risk_score']:.2f} ({risk_label})")
        print("    Top SHAP drivers:")
        for r in reasons:
            print(
                f"      • {r['feature_name']} ({r['impact_value']:+.2f}) "
                f"— {r['description']}"
            )
        print(f"    Action: {intervention['recommended_action']}")
        print(f"    Payload: {json.dumps(intervention['api_payload'])}")
        print()
