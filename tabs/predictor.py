# ─────────────────────────────────────────────
# tabs/predictor.py — Tab 4: Live Predictor
# ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pandas.api.types import is_numeric_dtype

from config import FRIENDLY_NAMES, PLOTLY_TEMPLATE


def render(
    models: dict,
    clean_df: pd.DataFrame,
    feat_names: list,
    threshold: float,
) -> None:
    st.markdown("### 🎯 Predecir Probabilidad de Default para un Nuevo Cliente")
    st.markdown(
        f"El umbral de decisión es **{threshold:.2f}** (configurable en la barra "
        "lateral). Ajusta los valores de abajo para que coincidan con el perfil "
        "del cliente."
    )

    # ── Input form ───────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    cols_cycle = [col_a, col_b, col_c]
    input_vals: dict = {}

    for i, feat in enumerate(feat_names):
        series = clean_df[feat]
        label  = FRIENDLY_NAMES.get(feat, feat)
        with cols_cycle[i % 3]:
            if is_numeric_dtype(series):
                c_min = float(series.min())
                c_max = float(series.max())
                c_med = float(series.median())
                step  = round(max((c_max - c_min) / 100, 0.01), 4)
                input_vals[feat] = st.number_input(
                    label, min_value=c_min, max_value=c_max,
                    value=c_med, step=step,
                    key=f"inp_{feat}",
                )
            else:
                options = sorted(series.dropna().astype(str).unique().tolist())
                mode_vals = series.mode(dropna=True)
                default = str(mode_vals.iloc[0]) if not mode_vals.empty else options[0]
                input_vals[feat] = st.selectbox(
                    label, options=options,
                    index=options.index(default) if default in options else 0,
                    key=f"inp_{feat}",
                )

    st.markdown("---")
    if not st.button("🔮 Predecir", type="primary", use_container_width=True):
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
    st.markdown("#### 📋 Resultados de la Predicción")

    p1, p2, p3 = st.columns(3)
    p1.metric("📈 Gradient Boosting", f"{gbc_prob:.1%}",
              delta="Default" if gbc_prob >= threshold else "Sin Default")
    p2.metric("🍃 LightGBM", f"{lgbm_prob:.1%}",
              delta="Default" if lgbm_prob >= threshold else "Sin Default")
    p3.metric("⚖️ Ensemble (promedio)", f"{avg_prob:.1%}")

    risk_cls   = "high-risk" if avg_prob >= threshold else "low-risk"
    risk_label = ("⚠️ ALTO RIESGO — Default probable"
                  if avg_prob >= threshold else
                  "✅ BAJO RIESGO — Default improbable")
    st.markdown(f'<span class="pred-badge {risk_cls}">{risk_label}</span>',
                unsafe_allow_html=True)

    # ── Gauge ────────────────────────────────────
    bar_color = "#ef4444" if avg_prob >= threshold else "#10b981"
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_prob * 100, 1),
        number={"suffix": "%"},
        delta={"reference": threshold * 100, "valueformat": ".1f"},
        title={"text": "Probabilidad de Default — Ensemble"},
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
    agreement = "Alto" if diff < 0.05 else "Moderado" if diff < 0.15 else "Bajo"
    color_map = {"Alto": "green", "Moderado": "orange", "Bajo": "red"}
    st.markdown(
        f"**Acuerdo entre Modelos:** :{color_map[agreement]}[{agreement}]  "
        f"(|GBC − LGBM| = {diff:.1%})"
    )
