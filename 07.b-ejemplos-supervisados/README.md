# Ejemplos supervisados (Scikit-learn)

Notebooks **completos** que aplican los conceptos del directorio [07-scikit-learn](../07-scikit-learn/): carga de CSV local, EDA, preprocesado con `Pipeline` + `ColumnTransformer`, comparación de varios modelos (incl. XGBoost y CatBoost) y métricas.

Todos comparten la **misma estructura de celdas**. El prefijo del nombre indica el tipo de problema:

| Prefijo | Tipo |
|---------|------|
| `01-regresion-lineal.ipynb` | **Plantilla** regresión (agnóstica; copiar para tu CSV) |
| `01-regresion-lineal-<dataset>.ipynb` | Ejemplos de regresión ya configurados |
| `02-clasificacion-binaria.ipynb` | **Plantilla** clasificación binaria |
| `02-clasificacion-binaria-<dataset>.ipynb` | Ejemplos binarios ya configurados |
| `03-clasificacion-multiple.ipynb` | **Plantilla** clasificación multiclase |
| `03-clasificacion-multiple-<dataset>.ipynb` | Ejemplos multiclase ya configurados |

Solo cambias el bloque **CONFIG** (y la exploración previa del CSV) si usas otro dataset.

## Requisitos

```bash
# Desde la raíz del repo
uv pip install -r requirements.txt
```

Incluye `xgboost` y `catboost` para la comparación completa (mismos imports que sklearn en `build_models()`). Para el examen puedes comentar o borrar líneas en `build_models()`.

Ejecuta Jupyter **desde esta carpeta** (`07.b-ejemplos-supervisados/`) para que las rutas `data/...` funcionen:

```bash
cd 07.b-ejemplos-supervisados
jupyter lab
```

## Preparación (una sola vez)

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

Esto crea en `data/`:

| Archivo | Usado por |
|---------|-----------|
| *(tu CSV en `data/`)* | **`01-regresion-lineal.ipynb`** (plantilla) |
| `wine_quality_red.csv` | `01-regresion-lineal-wine-quality-red.ipynb` |
| `auto_mpg.csv` | `01-regresion-lineal-auto-mpg.ipynb` |
| `diabetes.csv` | `01-regresion-lineal-diabetes.ipynb` |
| *(tu CSV)* | **`02-clasificacion-binaria.ipynb`** / **`03-clasificacion-multiple.ipynb`** |
| `breast_cancer.csv` | `02-clasificacion-binaria-breast-cancer.ipynb` |
| `bank_marketing.csv` | `02-clasificacion-binaria-bank-marketing.ipynb` |
| `iris.csv` | `03-clasificacion-multiple-iris.ipynb` |
| `wine_multiclass.csv` | `03-clasificacion-multiple-wine.ipynb` |

Más datasets alternativos y URLs en [data/README.md](data/README.md).

---

## Cómo se utiliza cada notebook

### Flujo común

Ejecuta las celdas **de arriba a abajo**. En plantillas y ejemplos recorres:

| Paso | Sección | Qué hace |
|------|---------|----------|
| 0 | **Helpers** | Funciones compartidas (imports, preprocesado, `build_models()` con sklearn + XGBoost + CatBoost). |
| 1 | **Explorar CSV** | Ver columnas, tipos y clases **antes** de CONFIG (plantillas y ejemplos). |
| 2 | **CONFIG** | Rutas, `TARGET_COL`, `DROP_COLS`, `build_models()`. |
| 3–5 | **Carga y EDA** | Lectura, faltantes, gráficos del target. |
| 6 | **Split** | `train_test_split` (estratificado en clasificación). |
| 7–8 | **Preprocesado y benchmark** | Mismo `ColumnTransformer` para todos los modelos. |
| 9 | **Mejor modelo** | Scatter (regresión) o matriz de confusión (clasificación). |

La primera ejecución completa puede tardar **varios minutos** (~8–9 modelos por notebook).

### Qué tocar para tu propio CSV

En la celda **CONFIG**, cambia como mínimo:

```python
DATA_PATH = "data/mi_archivo.csv"
CSV_SEP = ","              # ";" si el CSV usa punto y coma
TARGET_COL = "nombre_columna_objetivo"
DROP_COLS = ["id"]         # columnas que no deben entrar como features
```

Opcional, si la detección automática falla:

```python
FEATURE_COLS = ["col_a", "col_b", ...]
NUMERIC_COLS = [...]
CATEGORICAL_COLS = [...]
```

Para **añadir o quitar modelos**, edita la función `build_models()` (comenta o descomenta líneas en el diccionario `models`).

---

### [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) — plantilla base

**Empieza aquí** si tienes un CSV nuevo. Explica el flujo, las funciones y deja `DATA_PATH` / `TARGET_COL` como placeholders (`data/mi_dataset.csv`).

### [01-regresion-lineal-wine-quality-red.ipynb](01-regresion-lineal-wine-quality-red.ipynb) — ejemplo

Wine Quality Red: target `quality`, separador `;`.

### [01-regresion-lineal-auto-mpg.ipynb](01-regresion-lineal-auto-mpg.ipynb) — ejemplo

Auto MPG (UCI): target `mpg`, separador `,`, excluye `car_name`. El imputer del pipeline gestiona faltantes en `horsepower`.

### [01-regresion-lineal-diabetes.ipynb](01-regresion-lineal-diabetes.ipynb) — ejemplo

Diabetes: target `disease_progression`, separador `,`. Variables clínicas (edad, IMC, colesterol…); incluye faltantes (p. ej. en `bmi`).

**Métricas de la tabla:** MAE, RMSE y **R²** (se ordena por R², mayor es mejor).

**Modelos comparados:** `LinearRegression`, `Ridge`, `Lasso`, `RandomForest`, `GradientBoosting`, `HistGradientBoosting`, `XGBoost`, `CatBoost`.

**Resultado final:** gráfico *real vs predicho* del mejor modelo en el conjunto de test.

**Ejemplo — cambiar a otro CSV de regresión:**

```python
DATA_PATH = "data/mi_casas.csv"
CSV_SEP = ","
TARGET_COL = "precio"
DROP_COLS = ["id_vivienda"]
```

---

### [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) — plantilla base

**Empieza aquí** para clasificación **binaria** (dos clases). Placeholders en `data/mi_dataset.csv`.

### [02-clasificacion-binaria-breast-cancer.ipynb](02-clasificacion-binaria-breast-cancer.ipynb) — ejemplo

Breast Cancer: target `target` (0/1), excluye `diagnosis`.

### [02-clasificacion-binaria-bank-marketing.ipynb](02-clasificacion-binaria-bank-marketing.ipynb) — ejemplo

Bank Marketing: target `y` (yes/no), separador `;`, excluye `duration` (leakage).

**Métricas:** accuracy, precision, recall, **F1** (orden por F1). **Split** estratificado.

---

### [03-clasificacion-multiple.ipynb](03-clasificacion-multiple.ipynb) — plantilla base

**Empieza aquí** para **3+ clases**.

### [03-clasificacion-multiple-iris.ipynb](03-clasificacion-multiple-iris.ipynb) — ejemplo

Iris: target `species` (3 especies).

### [03-clasificacion-multiple-wine.ipynb](03-clasificacion-multiple-wine.ipynb) — ejemplo

Wine (sklearn): target `target` (3 cultivares), features numéricas.

**Métricas:** orden por **accuracy** (también F1 weighted en tabla).

---

## Resumen: ¿qué notebook abro?

| Tu problema | Notebook | Target típico |
|-------------|----------|----------------|
| Predecir un número (nuevo CSV) | **`01-regresion-lineal.ipynb`** | Columna numérica continua |
| Ver un ejemplo ya hecho | `01-regresion-lineal-wine-quality-red.ipynb`, `auto-mpg` o `diabetes` | — |
| Predecir sí/no (nuevo CSV) | **`02-clasificacion-binaria.ipynb`** | 2 clases |
| Ejemplo binario | `02-clasificacion-binaria-breast-cancer.ipynb` o `bank-marketing` | — |
| Predecir varias categorías (nuevo CSV) | **`03-clasificacion-multiple.ipynb`** | 3+ clases |
| Ejemplo multiclase | `03-clasificacion-multiple-iris.ipynb` o `wine` | — |

No hace falta tocar los notebooks 07.x: estos ejemplos **los aplican** en un flujo listo para copiar y adaptar.

## Cheat sheets relacionados

- [07.01 Fundamentos](../07-scikit-learn/07.01-scikit-learn-basics.ipynb)
- [07.02 Preprocesamiento](../07-scikit-learn/07.02-scikit-learn-preprocessing.ipynb)
- [07.03 Supervisado](../07-scikit-learn/07.03-scikit-learn-supervised-learning.ipynb)
- [07.05 Evaluación](../07-scikit-learn/07.05-scikit-learn-model-evaluation.ipynb)
- [07.06 Pipelines](../07-scikit-learn/07.06-scikit-learn-pipelines.ipynb)

## Carpeta `catboost_info/`

CatBoost crea por defecto esa carpeta con logs de entrenamiento al hacer `fit`. **No hace falta** para el notebook: en `build_models()` usamos `allow_writing_files=False`. Si ya se generó, puedes borrarla; está en `.gitignore`.

## Extensión opcional

Puedes añadir **LightGBM** en `build_models()` con un import directo (`from lightgbm import LGBMRegressor` / `LGBMClassifier`) y una entrada más en el diccionario `models`.
