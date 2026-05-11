import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label { color: #9aa0c0 !important; font-size: 0.78rem; }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important; font-size: 1.6rem; font-weight: 700;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px 8px 0 0;
        color: #9aa0c0; font-weight: 600; padding: 10px 22px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background: #13151f; border-right: 1px solid #2a2d3e; }
    [data-testid="stSidebar"] .stSlider { padding: 4px 0; }

    /* Section headers */
    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #c7d0f8;
        border-left: 4px solid #4f46e5; padding-left: 10px; margin: 20px 0 12px;
    }

    /* Prediction badge */
    .pred-badge {
        display: inline-block; padding: 8px 20px; border-radius: 20px;
        font-weight: 700; font-size: 1.1rem; margin-top: 8px;
    }
    .high-risk { background: #4a1a1a; color: #ff6b6b; border: 1px solid #ff4444; }
    .low-risk  { background: #1a3a2a; color: #51cf66; border: 1px solid #37b24d; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = ["#4f46e5", "#7c3aed", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"]

DROP_COLS   = ["Unnamed: 0", "n.credito"]
TARGET      = "default"
LABEL_MAP   = {0: "No Default", 1: "Default"}

FRIENDLY_NAMES = {
    "plazo": "Loan Term", "vinculacion": "Membership Duration",
    "v.cuota": "Installment Amount", "v.prestamo": "Loan Amount",
    "s.capital": "Capital Balance", "s.intereses": "Interest Balance",
    "aportes": "Contributions", "morosidad": "Delinquency Days",
    "garantias": "Guarantees", "valorgarantia": "Guarantee Value",
    "reestr": "Restructured", "ctasahorros": "Savings Accounts",
    "edad": "Age", "tipoasociado": "Member Type",
    "estado_cliente": "Client Status", "sexo": "Gender",
    "curtotalingresos": "Total Income", "curtotalegresos": "Total Expenses",
    "intestrato": "Income Stratum", "actualización": "Last Update",
    "puntaje_data": "Credit Score", "grupo_dptmto": "Dept. Group",
    "grupo_ciudad": "City Group", "grupo_edad": "Age Group",
    "grupo_actividadeco": "Activity Group", "cartera": "Portfolio",
    "actividadeconomica": "Economic Activity", "departamento": "Department",
    "ciudad": "City",
}


@st.cache_data(show_spinner=False)
def load_and_preprocess(file_obj) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (raw_df, clean_df)."""
    raw = pd.read_csv(file_obj)
    df  = raw.copy()

    # Drop ID columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Drop rows with nulls (only ~16 rows)
    df.dropna(inplace=True)

    # Encode categorical columns
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return raw, df, encoders, cat_cols


@st.cache_data(show_spinner=False)
def train_models(
    _df: pd.DataFrame,
    test_size: float,
    rf_n: int, rf_depth: int,
    xgb_n: int, xgb_depth: int, xgb_lr: float,
):
    X = _df.drop(columns=[TARGET])
    y = _df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    with st.spinner("Training Random Forest…"):
        rf = RandomForestRegressor(
            n_estimators=rf_n, max_depth=rf_depth,
            random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)

    with st.spinner("Training XGBoost…"):
        xgb = XGBRegressor(
            n_estimators=xgb_n, max_depth=xgb_depth,
            learning_rate=xgb_lr, random_state=42,
            eval_metric="logloss", verbosity=0
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)

    return rf, xgb, X_train, X_test, y_train, y_test, rf_pred, xgb_pred, X.columns.tolist()


def metrics(y_true, y_pred):
    return {
        "R²":   round(r2_score(y_true, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_true, y_pred)), 4),
        "MAE":  round(mean_absolute_error(y_true, y_pred), 4),
    }


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Default Predictor")
    st.markdown("---")

    uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])

    if uploaded is None:
        st.info("👆 Please upload `merged_df.csv` to continue.")
        st.stop()

    st.markdown("### ⚙️ Model Settings")
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05)

    with st.expander("🌲 Random Forest", expanded=True):
        rf_n     = st.slider("Trees (n_estimators)", 50, 300, 100, 25)
        rf_depth = st.slider("Max depth", 3, 20, 10, 1)

    with st.expander("⚡ XGBoost", expanded=True):
        xgb_n     = st.slider("Boosting rounds", 50, 300, 100, 25)
        xgb_depth = st.slider("Max depth ", 3, 12, 6, 1)
        xgb_lr    = st.slider("Learning rate", 0.01, 0.30, 0.10, 0.01)

    run_btn = st.button("🚀 Train Models", use_container_width=True, type="primary")

    st.markdown("---")
    st.caption("Models: Random Forest · XGBoost")
    st.caption("Task: Regression on default probability")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
raw_df, clean_df, encoders, cat_cols = load_and_preprocess(uploaded)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 💳 Credit Default Predictor")
st.markdown("Regression-based default probability using **Random Forest** and **XGBoost**")
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🔍 Exploratory Analysis",
    "🤖 Model Results",
    "🎯 Live Predictor",
])

# ══════════════════════════════════════════════
# TAB 1 — DATASET OVERVIEW
# ══════════════════════════════════════════════
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",     f"{len(raw_df):,}")
    c2.metric("Features",          f"{raw_df.shape[1]-3}")
    c3.metric("Default Rate",      f"{raw_df[TARGET].mean():.1%}")
    c4.metric("Missing Values",    f"{raw_df.isnull().sum().sum()}")

    st.markdown('<div class="section-title">Target Distribution</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 2])

    with col_a:
        vc = raw_df[TARGET].value_counts().reset_index()
        vc.columns = ["default", "count"]
        vc["label"] = vc["default"].map(LABEL_MAP)
        fig_pie = px.pie(
            vc, values="count", names="label",
            color_discrete_sequence=[COLORS[1], COLORS[0]],
            hole=0.55, template=PLOTLY_TEMPLATE,
        )
        fig_pie.update_traces(textposition="outside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        fig_bar = px.bar(
            vc, x="label", y="count",
            color="label", color_discrete_sequence=[COLORS[1], COLORS[0]],
            text="count", template=PLOTLY_TEMPLATE,
            labels={"label": "Class", "count": "Count"},
        )
        fig_bar.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig_bar.update_layout(showlegend=False, margin=dict(t=20, b=40, l=20, r=20), height=280)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-title">Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(
        raw_df.head(200).rename(columns=FRIENDLY_NAMES),
        use_container_width=True, height=300,
    )

    st.markdown('<div class="section-title">Descriptive Statistics</div>', unsafe_allow_html=True)
    num_cols = clean_df.select_dtypes(include="number").drop(columns=[TARGET]).columns
    st.dataframe(
        clean_df[num_cols].describe().T.style.format("{:.2f}").background_gradient(
            cmap="Blues", subset=["mean", "std"]
        ),
        use_container_width=True,
    )


# ══════════════════════════════════════════════
# TAB 2 — EXPLORATORY ANALYSIS
# ══════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Numeric Feature Distributions by Default Status</div>',
                unsafe_allow_html=True)

    key_num_cols = ["edad", "v.prestamo", "v.cuota", "s.capital", "morosidad",
                    "puntaje_data", "curtotalingresos", "curtotalegresos"]
    key_num_cols = [c for c in key_num_cols if c in raw_df.columns]

    sel_col = st.selectbox(
        "Select feature to explore",
        options=key_num_cols,
        format_func=lambda x: FRIENDLY_NAMES.get(x, x),
    )

    fig_hist = px.histogram(
        raw_df, x=sel_col, color=TARGET.replace("default", "default"),
        color_discrete_map={0: COLORS[0], 1: COLORS[1]},
        barmode="overlay", opacity=0.75, nbins=50,
        template=PLOTLY_TEMPLATE,
        labels={sel_col: FRIENDLY_NAMES.get(sel_col, sel_col), "default": "Default"},
        title=f"Distribution of {FRIENDLY_NAMES.get(sel_col, sel_col)} by Default Status",
    )
    fig_hist.update_layout(height=360, margin=dict(t=50, b=40, l=40, r=20))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)

    num_df = clean_df.select_dtypes(include="number")
    corr   = num_df.corr()
    renamed_corr = corr.rename(index=FRIENDLY_NAMES, columns=FRIENDLY_NAMES)

    fig_heat = go.Figure(go.Heatmap(
        z=renamed_corr.values,
        x=renamed_corr.columns.tolist(),
        y=renamed_corr.index.tolist(),
        colorscale="RdBu_r",
        zmid=0,
        hovertemplate="x: %{x}<br>y: %{y}<br>corr: %{z:.2f}<extra></extra>",
    ))
    fig_heat.update_layout(
        template=PLOTLY_TEMPLATE, height=520,
        margin=dict(t=20, b=20, l=20, r=20),
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-title">Correlation with Target (Default)</div>',
                unsafe_allow_html=True)

    target_corr = (
        num_df.corr()[TARGET]
        .drop(TARGET)
        .abs()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    target_corr.columns = ["Feature", "Absolute Correlation"]
    target_corr["Feature"] = target_corr["Feature"].map(lambda x: FRIENDLY_NAMES.get(x, x))

    fig_corr = px.bar(
        target_corr, x="Absolute Correlation", y="Feature",
        orientation="h", template=PLOTLY_TEMPLATE,
        color="Absolute Correlation", color_continuous_scale="Purples",
        title="Top 15 Features Correlated with Default",
    )
    fig_corr.update_layout(height=420, showlegend=False,
                           margin=dict(t=50, b=40, l=20, r=20),
                           yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown('<div class="section-title">Boxplots — Key Numeric Features</div>',
                unsafe_allow_html=True)

    box_col = st.selectbox(
        "Feature for boxplot",
        options=key_num_cols,
        format_func=lambda x: FRIENDLY_NAMES.get(x, x),
        key="box_sel",
    )
    fig_box = px.box(
        raw_df, x="default", y=box_col,
        color="default", color_discrete_map={0: COLORS[0], 1: COLORS[1]},
        template=PLOTLY_TEMPLATE,
        labels={"default": "Default", box_col: FRIENDLY_NAMES.get(box_col, box_col)},
        title=f"{FRIENDLY_NAMES.get(box_col, box_col)} by Default Status",
    )
    fig_box.update_layout(height=360, showlegend=False,
                          margin=dict(t=50, b=40, l=40, r=20))
    st.plotly_chart(fig_box, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ══════════════════════════════════════════════
with tab3:
    if not run_btn:
        st.info("👈 Adjust hyperparameters in the sidebar and click **Train Models** to see results.")
        st.stop()

    rf_model, xgb_model, X_train, X_test, y_train, y_test, rf_pred, xgb_pred, feat_names = \
        train_models(clean_df, test_size, rf_n, rf_depth, xgb_n, xgb_depth, xgb_lr)

    rf_m  = metrics(y_test, rf_pred)
    xgb_m = metrics(y_test, xgb_pred)

    # ── Metrics comparison ──────────────────────
    st.markdown('<div class="section-title">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🌲 Random Forest")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   rf_m["R²"])
        m2.metric("RMSE", rf_m["RMSE"])
        m3.metric("MAE",  rf_m["MAE"])

    with col2:
        st.markdown("#### ⚡ XGBoost")
        m1, m2, m3 = st.columns(3)
        m1.metric("R²",   xgb_m["R²"],
                  delta=round(xgb_m["R²"]  - rf_m["R²"],  4))
        m2.metric("RMSE", xgb_m["RMSE"],
                  delta=round(xgb_m["RMSE"] - rf_m["RMSE"], 4),
                  delta_color="inverse")
        m3.metric("MAE",  xgb_m["MAE"],
                  delta=round(xgb_m["MAE"]  - rf_m["MAE"],  4),
                  delta_color="inverse")

    # ── Radar chart ──────────────────────────────
    metrics_radar = ["R²", "1-RMSE", "1-MAE"]
    rf_vals  = [rf_m["R²"],  1-rf_m["RMSE"],  1-rf_m["MAE"]]
    xgb_vals = [xgb_m["R²"], 1-xgb_m["RMSE"], 1-xgb_m["MAE"]]

    fig_radar = go.Figure()
    for name, vals, color in [("Random Forest", rf_vals, COLORS[0]),
                               ("XGBoost",       xgb_vals, COLORS[1])]:
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=metrics_radar + [metrics_radar[0]],
            fill="toself", name=name,
            line_color=color, fillcolor=color, opacity=0.35,
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=PLOTLY_TEMPLATE, height=360,
        legend=dict(orientation="h", y=-0.15),
        margin=dict(t=40, b=60),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── Actual vs Predicted ───────────────────────
    st.markdown('<div class="section-title">Actual vs Predicted — Scatter</div>',
                unsafe_allow_html=True)

    model_choice = st.radio("Select model", ["Random Forest", "XGBoost"], horizontal=True)
    preds = rf_pred if model_choice == "Random Forest" else xgb_pred

    scatter_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": preds,
        "Error": np.abs(y_test.values - preds),
    })
    fig_scatter = px.scatter(
        scatter_df, x="Actual", y="Predicted", color="Error",
        color_continuous_scale="Plasma",
        template=PLOTLY_TEMPLATE,
        opacity=0.65, size_max=6,
        title=f"{model_choice} — Actual vs Predicted",
    )
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="white", dash="dash", width=1.5))
    fig_scatter.update_layout(height=400, margin=dict(t=50, b=40, l=40, r=20))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Residuals ────────────────────────────────
    st.markdown('<div class="section-title">Residuals Distribution</div>', unsafe_allow_html=True)

    residuals = y_test.values - preds
    fig_res = go.Figure()
    fig_res.add_trace(go.Histogram(
        x=residuals, nbinsx=60,
        marker_color=COLORS[0], opacity=0.8,
        name="Residuals",
    ))
    fig_res.add_vline(x=0, line_dash="dash", line_color="white", line_width=2)
    fig_res.update_layout(
        template=PLOTLY_TEMPLATE, height=320,
        title=f"{model_choice} Residuals",
        xaxis_title="Residual (Actual − Predicted)",
        yaxis_title="Count",
        margin=dict(t=50, b=40, l=40, r=20),
    )
    st.plotly_chart(fig_res, use_container_width=True)

    # ── Feature Importance ───────────────────────
    st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)

    top_n = st.slider("Show top N features", 5, 25, 15, 1)

    def feat_imp_fig(model, name, color):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
        fi_df["Feature"] = fi_df["Feature"].map(lambda x: FRIENDLY_NAMES.get(x, x))
        fi_df = fi_df.nlargest(top_n, "Importance").sort_values("Importance")
        fig = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            template=PLOTLY_TEMPLATE,
            color="Importance", color_continuous_scale="Purples" if "Forest" in name else "Blues",
            title=f"{name} — Top {top_n} Features",
        )
        fig.update_layout(height=50*top_n + 80, showlegend=False,
                          margin=dict(t=50, b=40, l=20, r=20))
        return fig

    col_fi1, col_fi2 = st.columns(2)
    with col_fi1:
        st.plotly_chart(feat_imp_fig(rf_model,  "Random Forest", COLORS[0]),
                        use_container_width=True)
    with col_fi2:
        st.plotly_chart(feat_imp_fig(xgb_model, "XGBoost",       COLORS[1]),
                        use_container_width=True)

    # ── Metrics bar comparison ────────────────────
    st.markdown('<div class="section-title">Side-by-Side Metric Comparison</div>',
                unsafe_allow_html=True)

    compare_df = pd.DataFrame({
        "Metric": ["R²", "RMSE", "MAE"],
        "Random Forest": [rf_m["R²"],   rf_m["RMSE"],   rf_m["MAE"]],
        "XGBoost":       [xgb_m["R²"],  xgb_m["RMSE"],  xgb_m["MAE"]],
    }).melt(id_vars="Metric", var_name="Model", value_name="Value")

    fig_compare = px.bar(
        compare_df, x="Metric", y="Value", color="Model", barmode="group",
        color_discrete_sequence=[COLORS[0], COLORS[1]],
        template=PLOTLY_TEMPLATE,
        text_auto=".4f",
        title="Metrics: Random Forest vs XGBoost",
    )
    fig_compare.update_layout(height=380, margin=dict(t=50, b=40, l=40, r=20))
    st.plotly_chart(fig_compare, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — LIVE PREDICTOR
# ══════════════════════════════════════════════
with tab4:
    if not run_btn:
        st.info("👈 Train the models first using the sidebar button.")
        st.stop()

    st.markdown("### 🎯 Predict Default Probability for a New Client")
    st.markdown("Adjust the sliders to set client attributes and get predictions from both models.")

    num_feature_cols = [c for c in feat_names if c != TARGET]

    col_a, col_b, col_c = st.columns(3)
    input_vals = {}
    cols = [col_a, col_b, col_c]

    for i, col in enumerate(num_feature_cols):
        c_min = float(clean_df[col].min())
        c_max = float(clean_df[col].max())
        c_med = float(clean_df[col].median())
        step  = max((c_max - c_min) / 100, 0.01)
        label = FRIENDLY_NAMES.get(col, col)
        with cols[i % 3]:
            input_vals[col] = st.number_input(
                label, min_value=c_min, max_value=c_max, value=c_med,
                step=round(step, 4), key=f"pred_{col}",
            )

    st.markdown("---")
    pred_btn = st.button("🔮 Predict Default Probability", type="primary", use_container_width=True)

    if pred_btn:
        input_df   = pd.DataFrame([input_vals])
        rf_prob    = float(np.clip(rf_model.predict(input_df)[0],  0, 1))
        xgb_prob   = float(np.clip(xgb_model.predict(input_df)[0], 0, 1))
        avg_prob   = (rf_prob + xgb_prob) / 2

        st.markdown("---")
        st.markdown("#### 📋 Prediction Results")

        p1, p2, p3 = st.columns(3)
        p1.metric("🌲 Random Forest",  f"{rf_prob:.1%}")
        p2.metric("⚡ XGBoost",        f"{xgb_prob:.1%}")
        p3.metric("⚖️ Ensemble (avg)", f"{avg_prob:.1%}")

        risk_class = "high-risk" if avg_prob >= 0.5 else "low-risk"
        risk_label = "⚠️ HIGH RISK — Default likely"  if avg_prob >= 0.5 else "✅ LOW RISK — Default unlikely"
        st.markdown(
            f'<span class="pred-badge {risk_class}">{risk_label}</span>',
            unsafe_allow_html=True,
        )

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(avg_prob * 100, 1),
            number={"suffix": "%"},
            delta={"reference": 50, "valueformat": ".1f"},
            title={"text": "Default Probability (Ensemble)"},
            gauge={
                "axis":  {"range": [0, 100]},
                "bar":   {"color": "#ef4444" if avg_prob >= 0.5 else "#10b981"},
                "steps": [
                    {"range": [0,  30], "color": "#1a3a2a"},
                    {"range": [30, 60], "color": "#3a3010"},
                    {"range": [60,100], "color": "#4a1a1a"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8, "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            template=PLOTLY_TEMPLATE, height=320,
            margin=dict(t=60, b=40, l=40, r=40),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Model agreement
        diff = abs(rf_prob - xgb_prob)
        agreement = "High" if diff < 0.05 else "Moderate" if diff < 0.15 else "Low"
        agree_color = {"High": "green", "Moderate": "orange", "Low": "red"}[agreement]
        st.markdown(
            f"**Model Agreement:** :{agree_color}[{agreement}]  "
            f"(|RF − XGB| = {diff:.1%})"
        )
