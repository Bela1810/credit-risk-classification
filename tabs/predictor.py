# ─────────────────────────────────────────────
# tabs/predictor.py — Tab 4: Live Predictor
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import FRIENDLY_NAMES, PLOTLY_TEMPLATE


def render(
    models: dict,
    clean_df: pd.DataFrame,
    feat_names: list,
    threshold: float,
) -> None:
    st.markdown("### 🎯 Predict Default Probability for a New Client")
    st.markdown(
        f"Decision threshold is **{threshold:.2f}** (set in sidebar). "
        "Adjust the inputs below to match a client profile."
    )

    # ── Input form ───────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    cols_cycle = [col_a, col_b, col_c]
    input_vals: dict = {}

    for i, feat in enumerate(feat_names):
        c_min = float(clean_df[feat].min())
        c_max = float(clean_df[feat].max())
        c_med = float(clean_df[feat].median())
        step  = round(max((c_max - c_min) / 100, 0.01), 4)
        label = FRIENDLY_NAMES.get(feat, feat)
        with cols_cycle[i % 3]:
            input_vals[feat] = st.number_input(
                label, min_value=c_min, max_value=c_max,
                value=c_med, step=step,
                key=f"inp_{feat}",
            )

    st.markdown("---")
    if not st.button("🔮 Predict", type="primary", use_container_width=True):
        return

    # ── Inference ────────────────────────────────
    input_df  = pd.DataFrame([input_vals])
    gbc_prob  = float(np.clip(
        models["Gradient Boosting"].predict_proba(input_df)[0, 1], 0, 1))
    lgbm_prob = float(np.clip(
        models["LightGBM"].predict_proba(input_df)[0, 1], 0, 1))
    avg_prob  = (gbc_prob + lgbm_prob) / 2

    # ── Results ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Prediction Results")

    p1, p2, p3 = st.columns(3)
    p1.metric("📈 Gradient Boosting", f"{gbc_prob:.1%}",
              delta="Default" if gbc_prob >= threshold else "No Default")
    p2.metric("🍃 LightGBM", f"{lgbm_prob:.1%}",
              delta="Default" if lgbm_prob >= threshold else "No Default")
    p3.metric("⚖️ Ensemble (avg)", f"{avg_prob:.1%}")

    risk_cls   = "high-risk" if avg_prob >= threshold else "low-risk"
    risk_label = ("⚠️ HIGH RISK — Default likely"
                  if avg_prob >= threshold else
                  "✅ LOW RISK — Default unlikely")
    st.markdown(f'<span class="pred-badge {risk_cls}">{risk_label}</span>',
                unsafe_allow_html=True)

    # ── Gauge ────────────────────────────────────
    bar_color = "#ef4444" if avg_prob >= threshold else "#10b981"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_prob * 100, 1),
        number={"suffix": "%"},
        delta={"reference": threshold * 100, "valueformat": ".1f"},
        title={"text": "Default Probability — Ensemble"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": bar_color},
            "steps": [
                {"range": [0,                   threshold * 60],  "color": "#1a3a2a"},
                {"range": [threshold * 60,       threshold * 100], "color": "#3a3010"},
                {"range": [threshold * 100, 100],                  "color": "#4a1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": threshold * 100,
            },
        },
    ))
    fig_gauge.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        margin=dict(t=60, b=40, l=40, r=40),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Model agreement ──────────────────────────
    diff      = abs(gbc_prob - lgbm_prob)
    agreement = "High" if diff < 0.05 else "Moderate" if diff < 0.15 else "Low"
    color_map = {"High": "green", "Moderate": "orange", "Low": "red"}
    st.markdown(
        f"**Model Agreement:** :{color_map[agreement]}[{agreement}]  "
        f"(|GBC − LGBM| = {diff:.1%})"
    )
