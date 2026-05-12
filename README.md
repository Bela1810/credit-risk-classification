# 📊 Credit Risk Classification – Cooperativa Financiera

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
├── data/                  # Dataset original y procesado
├── notebooks/            # Análisis exploratorio (EDA)
├── src/                  # Código fuente del modelo
├── models/               # Modelos entrenados (joblib)
├── reports/              # Resultados y métricas
├── requirements.txt      # Dependencias del proyecto
├── pyproject.toml        # Configuración del entorno (uv)
└── README.md

````

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
````

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

Entrenamiento del modelo:

```bash
python src/train.py
```

Ejecutar análisis exploratorio (EDA):

```bash
python notebooks/eda.py
```

---

## 📦 Guardado del modelo

El modelo entrenado se guarda en formato:

* **`.joblib`**

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








