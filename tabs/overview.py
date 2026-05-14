# ─────────────────────────────────────────────
# tabs/overview.py — Tab 1: Dataset Overview
# ─────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import streamlit as st

from config import COLORS, FRIENDLY_NAMES, PLOTLY_TEMPLATE, TARGET


def render(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    # ── KPI row ──────────────────────────────────
    n_features = max(clean_df.shape[1] - (1 if TARGET in clean_df.columns else 0), 0)
    default_rate = float(raw_df[TARGET].astype(int).mean()) if TARGET in raw_df.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",  f"{len(raw_df):,}")
    c2.metric("Features",       f"{n_features}")
    c3.metric("Default Rate",   f"{default_rate:.1%}")
    c4.metric("Missing Values", f"{int(raw_df.isnull().sum().sum()):,}")

    # ── Target distribution ──────────────────────
    st.markdown('<div class="section-title">Target Distribution</div>',
                unsafe_allow_html=True)

    vc = (
        raw_df[TARGET].astype(int).value_counts()
        .rename_axis("default").reset_index(name="count")
    )
    vc["label"] = vc["default"].map({0: "No Default", 1: "Default"})

    col_a, col_b = st.columns([1, 2])
    with col_a:
        fig_pie = px.pie(
            vc, values="count", names="label", hole=0.55,
            color_discrete_sequence=[COLORS[0], COLORS[4]],
            template=PLOTLY_TEMPLATE,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False,
                              margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            vc, x="label", y="count", color="label", text="count",
            color_discrete_sequence=[COLORS[0], COLORS[4]],
            template=PLOTLY_TEMPLATE,
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_bar.update_layout(showlegend=False,
                              margin=dict(t=20, b=40), height=280)
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Data sample ──────────────────────────────
    st.markdown('<div class="section-title">Data Sample</div>',
                unsafe_allow_html=True)
    st.dataframe(
        raw_df.head(200).rename(columns=FRIENDLY_NAMES),
        use_container_width=True, height=300,
    )

    # ── Descriptive statistics ───────────────────
    st.markdown('<div class="section-title">Descriptive Statistics</div>',
                unsafe_allow_html=True)
    num_cols = (
        clean_df.select_dtypes(include="number")
        .drop(columns=[TARGET], errors="ignore")
        .columns
    )
    st.dataframe(
        clean_df[num_cols].describe().T.style
        .format("{:.2f}")
        .background_gradient(cmap="Blues", subset=["mean", "std"]),
        use_container_width=True,
    )
