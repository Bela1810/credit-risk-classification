# ─────────────────────────────────────────────
# utils/loaders.py — data & model loading
# ─────────────────────────────────────────────
import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from config import DROP_COLS, TARGET, DEFAULT_DATA_PATH, MODEL_PATHS


@st.cache_resource(show_spinner="Loading models…")
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
            f"❌ Model file(s) not found: `{'`, `'.join(missing)}`\n\n"
            "Make sure the `.joblib` files are inside the `models/` folder."
        )
        st.stop()
    return models


@st.cache_data(show_spinner="Preprocessing data…")
def load_and_preprocess(
    file_obj,
    use_default: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw data from a parquet (default) or an uploaded CSV,
    then encode categoricals and return (raw_df, clean_df).
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

    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    return raw, df
