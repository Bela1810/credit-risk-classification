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
    "cartera":            "Tipo de Cartera",
    "plazo":              "Plazo del Préstamo (días)",
    "vinculacion":        "Antigüedad como Asociado (días)",
    "valor_cuota":        "Valor de la Cuota (COP)",
    "valor_prestamo":     "Monto del Préstamo (COP)",
    "saldo_capital":      "Saldo de Capital (COP)",
    "saldo_interes":      "Saldo de Intereses (COP)",
    "aportes":            "Aportes (COP)",
    "garantias":          "Tipo de Garantía",
    "valorgarantia":      "Valor de la Garantía (COP)",
    "reestr":             "Reestructurado",
    "ctasahorros":        "Saldo Cta. de Ahorros (COP)",
    "edad":               "Edad",
    "tipoasociado":       "Tipo de Asociado (1=Activo, 0=Inactivo)",
    "estado_cliente":     "Estado del Cliente",
    "departamento":       "Departamento",
    "sexo":               "Género (1=F, 0=M)",
    "curtotalingresos":   "Ingresos Totales (COP)",
    "curtotalegresos":    "Egresos Totales (COP)",
    "intestrato":         "Estrato Socioeconómico",
    "actualizacion":      "Datos Actualizados",
    "default":            "Default (1=Sí, 0=No)",
    "puntaje_data":       "Puntaje de Crédito",
    "grupo_dptmto":       "Grupo Departamento",
    "grupo_ciudad":       "Grupo Ciudad",
    "grupo_edad":         "Grupo Edad",
    "grupo_actividadeco": "Grupo Actividad Económica",
}

# Columns highlighted in EDA dropdowns
EDA_KEY_COLS = [
    "edad", "valor_prestamo", "valor_cuota", "saldo_capital",
    "saldo_interes", "puntaje_data", "curtotalingresos",
    "curtotalegresos", "valorgarantia", "ctasahorros",
    "intestrato", "grupo_edad",
]
