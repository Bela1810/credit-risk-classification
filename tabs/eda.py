# ─────────────────────────────────────────────
# tabs/eda.py — Tab 2: Exploratory Analysis
# ─────────────────────────────────────────────
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import COLORS, EDA_KEY_COLS, FRIENDLY_NAMES, PLOTLY_TEMPLATE, TARGET


def render(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    key_cols = [c for c in EDA_KEY_COLS if c in raw_df.columns]

    # ── Histogram by default status ──────────────
    st.markdown('<div class="section-title">Feature Distribution by Default Status</div>',
                unsafe_allow_html=True)
    sel_col = st.selectbox(
        "Select feature",
        key_cols,
        format_func=lambda x: FRIENDLY_NAMES.get(x, x),
    )
    fig_hist = px.histogram(
        raw_df, x=sel_col, color="default",
        color_discrete_map={0: COLORS[0], 1: COLORS[4]},
        barmode="overlay", opacity=0.75, nbins=50,
        template=PLOTLY_TEMPLATE,
        labels={sel_col: FRIENDLY_NAMES.get(sel_col, sel_col), "default": "Default"},
        title=f"{FRIENDLY_NAMES.get(sel_col, sel_col)} — by Default Status",
    )
    fig_hist.update_layout(height=360, margin=dict(t=50, b=40, l=40, r=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    # ── Correlation with target + Boxplot ────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Correlation with Target</div>',
                    unsafe_allow_html=True)
        num_df = clean_df.select_dtypes(include="number")
        if TARGET not in num_df.columns:
            num_df = num_df.copy()
            num_df[TARGET] = clean_df[TARGET].astype(int)

        target_corr = (
            num_df.corr()[TARGET]
            .drop(TARGET, errors="ignore").abs()
            .sort_values(ascending=False).head(15).reset_index()
        )
        target_corr.columns = ["Feature", "Abs. Correlation"]
        target_corr["Feature"] = target_corr["Feature"].map(
            lambda x: FRIENDLY_NAMES.get(x, x))

        fig_corr = px.bar(
            target_corr, x="Abs. Correlation", y="Feature", orientation="h",
            color="Abs. Correlation", color_continuous_scale="Purples",
            template=PLOTLY_TEMPLATE,
            title="Top 15 Features Correlated with Default",
        )
        fig_corr.update_layout(height=430, showlegend=False,
                               yaxis=dict(autorange="reversed"),
                               margin=dict(t=50, b=40, l=20, r=20))
        st.plotly_chart(fig_corr, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Boxplot by Default Status</div>',
                    unsafe_allow_html=True)
        box_col = st.selectbox(
            "Feature for boxplot", key_cols,
            format_func=lambda x: FRIENDLY_NAMES.get(x, x),
            key="box_sel",
        )
        fig_box = px.box(
            raw_df, x="default", y=box_col,
            color="default", color_discrete_map={0: COLORS[0], 1: COLORS[4]},
            template=PLOTLY_TEMPLATE,
            labels={"default": "Default",
                    box_col: FRIENDLY_NAMES.get(box_col, box_col)},
        )
        fig_box.update_layout(height=400, showlegend=False,
                              margin=dict(t=50, b=40, l=40, r=20))
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Correlation heatmap ──────────────────────
    st.markdown('<div class="section-title">Correlation Heatmap</div>',
                unsafe_allow_html=True)
    num_for_heat = clean_df.select_dtypes(include="number")
    if TARGET not in num_for_heat.columns:
        num_for_heat = num_for_heat.copy()
        num_for_heat[TARGET] = clean_df[TARGET].astype(int)

    corr         = num_for_heat.corr()
    renamed_corr = corr.rename(index=FRIENDLY_NAMES, columns=FRIENDLY_NAMES)

    fig_heat = go.Figure(go.Heatmap(
        z=renamed_corr.values,
        x=renamed_corr.columns.tolist(),
        y=renamed_corr.index.tolist(),
        colorscale="RdBu_r", zmid=0,
        hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE, height=520,
        xaxis_tickangle=-45,
        margin=dict(t=20, b=60, l=20, r=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)
