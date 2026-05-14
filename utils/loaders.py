# ─────────────────────────────────────────────
# utils/loaders.py — data & model loading
# ─────────────────────────────────────────────
import joblib
import pandas as pd
import streamlit as st

from config import DROP_COLS, TARGET, DEFAULT_DATA_PATH, MODEL_PATHS


@st.cache_resource(show_spinner="Cargando modelos…")
def load_models() -> dict:
    """Load all joblib models. Stops the app with a clear error if any file is missing."""
    models, missing = {}, []
    for name, path in MODEL_PATHS.items():
        try:
            models[name] = joblib.load(path)
        except FileNotFoundError:
            missing.append(path)
    if missing:
        st.error(
            f"❌ Archivo(s) de modelo no encontrado(s): `{'`, `'.join(missing)}`\n\n"
            "Asegúrate de que los archivos `.joblib` estén dentro de la carpeta `models/`."
        )
        st.stop()
    return models


@st.cache_data(show_spinner="Procesando datos…")
def load_and_preprocess(
    file_obj,
    use_default: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data from a parquet (default) or an uploaded CSV,
    then normalize dtypes and return (raw_df, clean_df).

    Note: the saved model pipelines already include a ColumnTransformer
    with OneHotEncoder for nominal columns, so we keep categoricals as
    plain strings rather than label-encoding them here.
    """
    if use_default:
        raw = pd.read_parquet(DEFAULT_DATA_PATH)
    else:
        raw = pd.read_csv(file_obj)

    df = raw.copy()
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)
    df.dropna(inplace=True)

    # Ensure target is always int (parquet may store as bool / int8)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(int)

    # Convert pandas Categorical columns (common in parquet) to plain
    # string dtype so downstream code (.min/.max/.median, sklearn,
    # plotly) doesn't trip on unordered Categoricals.
    for col in df.select_dtypes(include=["category"]).columns:
        df[col] = df[col].astype(str)

    return raw, df
