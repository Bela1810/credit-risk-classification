# ─────────────────────────────────────────────
# config.py — all project-wide constants
# ─────────────────────────────────────────────

PLOTLY_TEMPLATE   = "plotly_dark"
COLORS            = ["#4f46e5", "#10b981", "#06b6d4", "#f59e0b", "#ef4444", "#7c3aed"]

TARGET            = "default"
DROP_COLS         = ["Unnamed: 0", "n.credito"]
DEFAULT_DATA_PATH = "data/processed/02_datos_ajustados_cooperativa.parquet"

MODEL_PATHS = {
    "Gradient Boosting": "models/gbc_best_model.joblib",
    "LightGBM":          "models/lightgbm_best_model.joblib",
}

MODEL_COLORS = {
    "Gradient Boosting": COLORS[0],
    "LightGBM":          COLORS[1],
}

FRIENDLY_NAMES = {
    "cartera":            "Portfolio Type",
    "plazo":              "Loan Term (days)",
    "vinculacion":        "Membership Duration (days)",
    "valor_cuota":        "Installment Amount (COP)",
    "valor_prestamo":     "Loan Amount (COP)",
    "saldo_capital":      "Capital Balance (COP)",
    "saldo_interes":      "Interest Balance (COP)",
    "aportes":            "Contributions (COP)",
    "garantias":          "Guarantee Type",
    "valorgarantia":      "Guarantee Value (COP)",
    "reestr":             "Restructured",
    "ctasahorros":        "Savings Account Balance (COP)",
    "edad":               "Age",
    "tipoasociado":       "Member Type (1=Active, 0=Inactive)",
    "estado_cliente":     "Client Status",
    "departamento":       "Department",
    "sexo":               "Gender (1=F, 0=M)",
    "curtotalingresos":   "Total Income (COP)",
    "curtotalegresos":    "Total Expenses (COP)",
    "intestrato":         "Socioeconomic Stratum",
    "actualizacion":      "Data Updated",
    "default":            "Default (1=Yes, 0=No)",
    "puntaje_data":       "Credit Score",
    "grupo_dptmto":       "Department Group",
    "grupo_ciudad":       "City Group",
    "grupo_edad":         "Age Group",
    "grupo_actividadeco": "Economic Activity Group",
}

# Columns highlighted in EDA dropdowns
EDA_KEY_COLS = [
    "edad", "valor_prestamo", "valor_cuota", "saldo_capital",
    "saldo_interes", "puntaje_data", "curtotalingresos",
    "curtotalegresos", "valorgarantia", "ctasahorros",
    "intestrato", "grupo_edad",
]
