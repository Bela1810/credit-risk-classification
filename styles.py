# ─────────────────────────────────────────────
# styles.py — CSS injection
# ─────────────────────────────────────────────
import streamlit as st

CSS = """
<style>
    .stApp { background-color: #0f1117; }

    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e2130, #252840);
        border: 1px solid #3a3f5c;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label {
        color: #9aa0c0 !important; font-size: 0.78rem;
    }
    div[data-testid="metric-container"] div[data-testid="metric-value"] {
        color: #ffffff !important; font-size: 1.6rem; font-weight: 700;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px 8px 0 0;
        color: #9aa0c0; font-weight: 600; padding: 10px 22px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }

    [data-testid="stSidebar"] {
        background: #13151f; border-right: 1px solid #2a2d3e;
    }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #c7d0f8;
        border-left: 4px solid #4f46e5; padding-left: 10px;
        margin: 20px 0 12px;
    }
    .model-badge {
        display: inline-block; padding: 4px 12px; border-radius: 20px;
        font-weight: 700; font-size: 0.8rem; margin-bottom: 6px;
    }
    .badge-gbc  { background: #1e2a4a; color: #818cf8; border: 1px solid #4f46e5; }
    .badge-lgbm { background: #1a3a2a; color: #34d399; border: 1px solid #10b981; }

    .pred-badge {
        display: inline-block; padding: 10px 24px; border-radius: 20px;
        font-weight: 700; font-size: 1.15rem; margin-top: 10px;
    }
    .high-risk { background: #4a1a1a; color: #ff6b6b; border: 1px solid #ff4444; }
    .low-risk  { background: #1a3a2a; color: #51cf66; border: 1px solid #37b24d; }
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
