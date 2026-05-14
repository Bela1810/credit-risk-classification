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
from utils.metrics import compute_probs, evaluate_all, get_pipeline_input_features

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be the very first st call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor de Default Crediticio",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 Predictor de Default Crediticio")
    st.markdown("---")

    st.markdown("### 📂 Fuente de Datos")
    data_source = st.radio(
        "Elige la fuente de datos",
        ["📁 Usar dataset por defecto", "⬆️ Subir un archivo CSV"],
        help="El dataset por defecto se carga desde el directorio `data/` del repositorio",
    )

    uploaded    = None
    use_default = data_source == "📁 Usar dataset por defecto"

    if use_default:
        if not os.path.exists(DEFAULT_DATA_PATH):
            st.error(
                f"❌ Archivo por defecto no encontrado: `{DEFAULT_DATA_PATH}`\n\n"
                "Asegúrate de que `02_datos_ajustados_cooperativa.parquet` "
                "esté dentro de la carpeta `data/`."
            )
            st.stop()
        st.success("✅ Usando dataset por defecto")
    else:
        uploaded = st.file_uploader("Sube un dataset (CSV)", type=["csv"])
        if uploaded is None:
            st.info("👆 Sube un archivo CSV para continuar.")
            st.stop()

    st.markdown("### 🔧 Configuración de Evaluación")
    test_size = st.slider("Tamaño del conjunto de prueba", 0.10, 0.40, 0.20, 0.05)
    threshold = st.slider("Umbral de decisión", 0.10, 0.90, 0.50, 0.05,
                          help="Umbral de probabilidad para clasificar como Default. "
                               "Bajarlo aumenta el Recall (detecta más defaults) a costa "
                               "de más falsos positivos.")

    st.markdown("---")
    st.markdown(
        '<span class="model-badge badge-gbc">Gradient Boosting</span><br>'
        '<span class="model-badge badge-lgbm">LightGBM</span>',
        unsafe_allow_html=True,
    )
    st.caption("Modelos pre-entrenados cargados desde `models/`")

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

probs       = compute_probs(loaded_models, X_test, tuple(loaded_models.keys()))
results     = evaluate_all(probs, y_test, threshold)
model_names = list(results.keys())
feat_names  = X.columns.tolist()

# Features the saved sklearn pipelines actually consume (union across models).
# Used by the Live Predictor so we don't ask the user for inputs the model ignores.
_model_feats: list = []
_seen: set = set()
for _m in loaded_models.values():
    used = get_pipeline_input_features(_m)
    if not used:
        continue
    for f in used:
        if f not in _seen and f in clean_df.columns:
            _model_feats.append(f)
            _seen.add(f)
predictor_feats = _model_feats or feat_names

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# 💳 Predictor de Default Crediticio")
st.markdown(
    "Detección de incumplimiento basada en clasificación, usando los modelos "
    "pre-entrenados **Gradient Boosting** y **LightGBM**"
)
st.markdown("---")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen del Dataset",
    "🔍 Análisis Exploratorio",
    "🤖 Resultados de los Modelos",
    "🎯 Predictor en Vivo",
])

with tab1:
    overview.render(raw_df, clean_df)

with tab2:
    eda.render(raw_df, clean_df)

with tab3:
    models.render(loaded_models, results, model_names, feat_names, y_test, threshold)

with tab4:
    predictor.render(loaded_models, clean_df, predictor_feats, threshold)
