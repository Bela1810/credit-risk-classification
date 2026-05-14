# 📊 Credit Risk Classification – Cooperativa Financiera

## Integrantes

- Isabella Montoya
- Daniel Ospina
- Juan Camilo Manjarres

## 📌 Descripción del proyecto

Este proyecto tiene como objetivo el desarrollo de un modelo de **clasificación de riesgo crediticio** basado en datos históricos de una cooperativa financiera.

El dataset contiene información consolidada de créditos otorgados a los asociados, incluyendo variables financieras, socioeconómicas, geográficas y de comportamiento con la cooperativa.

La variable objetivo es:

- **`default`**
  - `1`: El cliente incurrió en impago (default)
  - `0`: El cliente se encuentra al día con sus obligaciones

El propósito principal del análisis es **identificar patrones asociados al incumplimiento de pago**, para posteriormente construir un modelo predictivo que apoye la toma de decisiones en la aprobación y seguimiento de créditos.

---

## 📁 Estructura del proyecto

```

credit-risk-classification/
│
├── app.py                 # Entrada principal (Streamlit)
├── config.py              # Constantes globales (paths, nombres bonitos, colores)
├── styles.py              # CSS inyectado en la app
├── tabs/                  # Una pestaña por archivo (overview, eda, models, predictor)
├── utils/                 # Loaders + helpers de métricas
├── data/                  # Dataset original y procesado (.parquet)
├── models/                # Modelos entrenados (.joblib)
├── notebooks/             # Análisis y entrenamiento (Jupyter)
├── pyproject.toml         # Configuración del entorno (uv)
├── requirements.txt       # Dependencias del proyecto
└── README.md

```

---

## ⚙️ Requisitos

- Python 3.10 o superior
- uv (gestor de entornos virtuales y dependencias)

---

## 🚀 Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd credit-risk-classification
```

### 2. Instalar dependencias

Instalar uv (si no lo tienes):

```bash
pip install uv
```

Sincronizar el entorno:

```bash
uv sync
```

---

### 3. Activar el entorno virtual

**Windows (PowerShell):**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

---

## 🧠 Ejecución del proyecto

Lanzar la aplicación interactiva (Streamlit):

```bash
streamlit run app.py
```

La app expone cuatro pestañas:

1. **Dataset Overview** — KPIs, distribución del target y estadísticas descriptivas.
2. **Exploratory Analysis** — histogramas, boxplots, correlaciones y heatmap.
3. **Model Results** — métricas, ROC / PR, matrices de confusión, importancia de variables.
4. **Live Predictor** — formulario para predecir la probabilidad de default de un nuevo cliente.

Reentrenamiento (notebooks Jupyter en orden):

```bash
jupyter lab notebooks/
# 01_descarga_datos.ipynb
# 02_ajuste_tratamiento_datos.ipynb
# 04.AutoML.ipynb
# 05_entrenamiento_modelo.ipynb
# 06_interpretación_modelos.ipynb
```

Los modelos entrenados se guardan en `models/` como archivos `.joblib` y son cargados automáticamente por la app.

---

## 📦 Guardado del modelo

El modelo entrenado se guarda en formato:

- **`.joblib`**

Este formato permite reutilizar el modelo de forma eficiente para predicciones futuras.

---

## 🧠 Metodología

1. Limpieza y preparación de datos
2. Análisis exploratorio (EDA)
3. Realización de AutoML usando herramientas como PyCaret
4. Entrenamiento de modelos de clasificación
5. Evaluación del desempeño
6. Guardado del modelo final
7. Desarrollo del modelo y visualización usando Streamlit

---
