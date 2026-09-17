"""
app.py — Autonomous AI Retention Agent Dashboard
==================================================
Streamlit frontend for the AI Customer Churn Retention Agent.
All data is sourced exclusively from mock_data.py.
"""

import streamlit as st
import plotly.graph_objects as go

from agent_engine import get_real_agent_intervention

from train_model import (
    get_real_customer_list,
    get_real_churn_prediction,
    get_real_shap_reasons,
)

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Autonomous Retention Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ---------- Header ---------- */
    .main-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        padding: 1.8rem 2.2rem;
        border-radius: 16px;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }
    .main-header h1 {
        margin: 0;
        font-size: 1.85rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.35rem 0 0 0;
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 400;
    }

    /* ---------- Section Cards ---------- */
    .section-card {
        background: linear-gradient(145deg, #1e1b4b 0%, #1a1a2e 100%);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .section-card:hover {
        border-color: rgba(139, 92, 246, 0.35);
        box-shadow: 0 6px 28px rgba(139, 92, 246, 0.1);
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #c4b5fd;
        margin-bottom: 1rem;
        letter-spacing: 0.02em;
    }

    /* ---------- Metric Chips ---------- */
    .metric-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.65rem;
    }
    .metric-chip {
        background: rgba(30, 27, 75, 0.7);
        border: 1px solid rgba(139, 92, 246, 0.12);
        border-radius: 10px;
        padding: 0.7rem 0.9rem;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-chip:hover {
        transform: translateY(-2px);
        border-color: rgba(139, 92, 246, 0.35);
    }
    .metric-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748b;
        letter-spacing: 0.08em;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        font-size: 1.05rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* ---------- Risk Badge ---------- */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.7rem 1.2rem;
        border-radius: 12px;
        font-size: 1.3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .risk-high {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #f87171;
    }
    .risk-low {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #4ade80;
    }
    .risk-label-text {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* ---------- Warning Banner ---------- */
    .warning-banner {
        background: linear-gradient(90deg, rgba(239,68,68,0.08) 0%, rgba(239,68,68,0.02) 100%);
        border-left: 4px solid #ef4444;
        border-radius: 0 10px 10px 0;
        padding: 0.75rem 1rem;
        margin-top: 0.8rem;
        color: #fca5a5;
        font-size: 0.82rem;
        font-weight: 500;
    }

    /* ---------- Agent Result Card ---------- */
    .agent-result {
        background: linear-gradient(145deg, #064e3b 0%, #022c22 100%);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 14px;
        padding: 1.5rem 1.6rem;
        margin-top: 0.8rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }
    .agent-result-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #6ee7b7;
        margin-bottom: 1rem;
    }
    .sub-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }
    .crm-note {
        background: rgba(30, 27, 75, 0.5);
        border: 1px solid rgba(139, 92, 246, 0.15);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        color: #cbd5e1;
        font-size: 0.88rem;
        line-height: 1.5;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #c4b5fd !important;
        font-weight: 600;
    }

    /* ---------- Button ---------- */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 50%, #818cf8 100%);
        color: white;
        font-weight: 700;
        font-size: 0.95rem;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        letter-spacing: 0.02em;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }
    .stButton > button:active {
        transform: translateY(0);
    }

    /* ---------- Plotly Chart Container ---------- */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- SHAP description text ---------- */
    .shap-insight {
        background: rgba(99, 102, 241, 0.06);
        border-left: 3px solid #818cf8;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.9rem;
        margin-top: 0.6rem;
        color: #a5b4fc;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>⚡ Autonomous Retention Agent (SaaS &amp; Telco)</h1>
        <p>AI-powered churn prediction, explainability, and autonomous retention actions</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar – Customer Selection ──────────────────────────────────────────────
customers = get_real_customer_list()
customer_ids = [c["customer_id"] for c in customers]
customer_lookup = {c["customer_id"]: c for c in customers}

with st.sidebar:
    st.markdown("### 👤 Customer Selection")
    st.markdown("---")
    selected_id = st.selectbox(
        "Select Customer ID",
        customer_ids,
        index=0,
        help="Choose a customer to analyse their churn risk and generate a retention strategy.",
    )
    st.markdown("---")
    st.markdown(
        f"<div style='text-align:center; color:#64748b; font-size:0.75rem;'>"
        f"Showing <b style='color:#c4b5fd;'>{len(customers)}</b> customers in database"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Fetch data for selected customer ──────────────────────────────────────────
customer = customer_lookup[selected_id]
prediction = get_real_churn_prediction(selected_id)
shap_reasons = get_real_shap_reasons(selected_id)

risk_score = prediction["churn_risk_score"]
is_high_risk = prediction["is_high_risk"]

# ── Main Layout ───────────────────────────────────────────────────────────────
col1, col2 = st.columns([0.4, 0.6], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Profile & Risk Engine
# ══════════════════════════════════════════════════════════════════════════════
with col1:
    # ── Customer Profile ──────────────────────────────────────────────────
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">📋 Customer Profile</div>'
        '<div class="metric-grid">'
        f'<div class="metric-chip"><div class="metric-label">Customer ID</div>'
        f'<div class="metric-value">{customer["customer_id"]}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Contract</div>'
        f'<div class="metric-value">{customer["Contract"]}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Tenure</div>'
        f'<div class="metric-value">{customer["tenure"]} months</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Monthly Charges</div>'
        f'<div class="metric-value">${customer["MonthlyCharges"]:.2f}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Total Charges</div>'
        f'<div class="metric-value">${customer["TotalCharges"]:,.2f}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Payment Method</div>'
        f'<div class="metric-value">{customer["PaymentMethod"]}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Tech Support</div>'
        f'<div class="metric-value">{"✅ Yes" if customer["TechSupport"] == "Yes" else "❌ No"}</div></div>'
        f'<div class="metric-chip"><div class="metric-label">Paperless Billing</div>'
        f'<div class="metric-value">{"✅ Yes" if customer["PaperlessBilling"] == "Yes" else "❌ No"}</div></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

    # ── Churn Risk Score ──────────────────────────────────────────────────
    risk_pct = f"{risk_score * 100:.0f}%"
    risk_class = "risk-high" if is_high_risk else "risk-low"
    risk_icon = "🔴" if is_high_risk else "🟢"
    risk_text = "HIGH RISK" if is_high_risk else "LOW RISK"

    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">🎯 Churn Risk Score</div>'
        f'<div class="risk-badge {risk_class}">'
        f'{risk_icon} {risk_pct}'
        f'<span class="risk-label-text">{risk_text}</span>'
        "</div>"
        + (
            '<div class="warning-banner">'
            "⚠️ This customer is flagged as <b>high churn risk</b>. "
            "Immediate retention action is recommended."
            "</div>"
            if is_high_risk
            else ""
        )
        + "</div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — SHAP Explainability + Agent Action
# ══════════════════════════════════════════════════════════════════════════════
with col2:
    # ── SHAP Feature Impact Chart ─────────────────────────────────────────
    st.markdown(
        '<div class="section-card">'
        '<div class="section-title">🔍 Why is this customer at risk?</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    # Prepare data (sorted ascending so the highest impact is at the top of horizontal bar)
    feature_names = [r["feature_name"] for r in reversed(shap_reasons)]
    impact_values = [r["impact_value"] for r in reversed(shap_reasons)]

    # Colour gradient based on impact
    bar_colors = [
        f"rgba({min(255, int(130 + v * 200))}, {max(50, int(140 - v * 150))}, {max(80, int(246 - v * 300))}, 0.85)"
        for v in impact_values
    ]

    fig = go.Figure(
        go.Bar(
            x=impact_values,
            y=feature_names,
            orientation="h",
            marker=dict(
                color=bar_colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{v:.2f}" for v in impact_values],
            textposition="outside",
            textfont=dict(color="#c4b5fd", size=13, family="Inter"),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Impact: <b>%{x:.3f}</b><br>"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#94a3b8"),
        xaxis=dict(
            title="SHAP Impact Value",
            title_font=dict(size=12, color="#64748b"),
            gridcolor="rgba(148,163,184,0.08)",
            zeroline=False,
            range=[0, max(impact_values) * 1.35] if impact_values else [0, 1],
        ),
        yaxis=dict(
            tickfont=dict(size=13, color="#e2e8f0"),
            gridcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=10, r=40, t=10, b=40),
        height=220,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # SHAP insight text
    top_driver = shap_reasons[0] if shap_reasons else None
    if top_driver:
        st.markdown(
            f'<div class="shap-insight">'
            f'<b>Top Driver:</b> <code>{top_driver["feature_name"]}</code> '
            f'(impact: {top_driver["impact_value"]:.2f}) — '
            f'{top_driver["description"]}'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Agent Intervention Button ─────────────────────────────────────────
    run_agent = st.button("🚀 Run Autonomous Retention Agent", use_container_width=True)

    if run_agent:
        with st.spinner("Agent reasoning in progress..."):
            try:
                intervention = get_real_agent_intervention(
                    customer, risk_score, shap_reasons
                )
            except Exception as e:
                st.error(f"LLM API Error: {e}")
                st.info("Ensure your Groq API key is valid and you have access to the specified model.")
                st.stop()

        st.markdown(
            '<div class="agent-result">'
            '<div class="agent-result-title">✅ Retention Strategy Generated</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        # Email Draft
        st.markdown('<div class="sub-label">📧 Generated Email Draft</div>', unsafe_allow_html=True)
        st.text_area(
            "Email Draft",
            value=intervention["generated_email_draft"],
            height=180,
            label_visibility="collapsed",
            key="email_draft_area",
        )

        # CRM Note
        st.markdown('<div class="sub-label">📝 Recommended Action (CRM Note)</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="crm-note">{intervention["recommended_action"]}</div>',
            unsafe_allow_html=True,
        )

        # Removed API payload per hackathon scope constraints
