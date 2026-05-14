# ─────────────────────────────────────────────
# app.py — entry point
# ─────────────────────────────────────────────
import os
import warnings

import streamlit as st
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── Project modules ──────────────────────────
from config import DEFAULT_DATA_PATH, TARGET
from styles import inject_css
from tabs import eda, models, overview, predictor
from utils.loaders import load_and_preprocess, load_models
from utils.metrics import evaluate_all

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Credit Default Predictor")
    st.markdown("---")

    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose data source",
        ["📁 Use default dataset", "⬆️ Upload a CSV file"],
        help="Default dataset is loaded from the `data/` directory in the repo",
    )

    uploaded    = None
    use_default = data_source == "📁 Use default dataset"

    if use_default:
        if not os.path.exists(DEFAULT_DATA_PATH):
            st.error(
                f"❌ Default file not found: `{DEFAULT_DATA_PATH}`\n\n"
                "Make sure `02_datos_ajustados_cooperativa.parquet` "
                "is inside the `data/` folder."
            )
            st.stop()
        st.success("✅ Using default dataset")
    else:
        uploaded = st.file_uploader("Upload dataset (CSV)", type=["csv"])
        if uploaded is None:
            st.info("👆 Upload a CSV file to continue.")
            st.stop()

    st.markdown("### 🔧 Evaluation Settings")
    test_size = st.slider("Test set size",      0.10, 0.40, 0.20, 0.05)
    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05,
                          help="Probability cutoff for classifying as Default")

    st.markdown("---")
    st.markdown(
        '<span class="model-badge badge-gbc">Gradient Boosting</span><br>'
        '<span class="model-badge badge-lgbm">LightGBM</span>',
        unsafe_allow_html=True,
    )
    st.caption("Pre-trained models loaded from `models/`")

# ─────────────────────────────────────────────
# LOAD DATA & MODELS
# ─────────────────────────────────────────────
loaded_models    = load_models()
raw_df, clean_df = load_and_preprocess(uploaded, use_default=use_default)

X = clean_df.drop(columns=[TARGET], errors="ignore")
y = clean_df[TARGET].astype(int)

_, X_test, _, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y,
)

results     = evaluate_all(loaded_models, X_test, y_test, threshold)
model_names = list(results.keys())
feat_names  = X.columns.tolist()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 💳 Credit Default Predictor")
st.markdown(
    "Classification-based default detection using pre-trained "
    "**Gradient Boosting** and **LightGBM** models"
)
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

with tab1:
    overview.render(raw_df, clean_df)

with tab2:
    eda.render(raw_df, clean_df)

with tab3:
    models.render(loaded_models, results, model_names, feat_names, y_test, threshold)

with tab4:
    predictor.render(loaded_models, clean_df, feat_names, threshold)
