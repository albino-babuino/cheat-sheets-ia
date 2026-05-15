# Ejemplos supervisados (Scikit-learn)

Notebooks **completos** que aplican los conceptos del directorio [07-scikit-learn](../07-scikit-learn/): carga de CSV local, EDA, preprocesado con `Pipeline` + `ColumnTransformer`, split **train / validación / test**, comparación de modelos (sklearn + XGBoost + CatBoost) y métricas.

## Convención de nombres

| Prefijo | Qué es |
|---------|--------|
| `01-regresion-lineal.ipynb` | **Plantilla** de regresión (raíz de esta carpeta) |
| `02-clasificacion-binaria.ipynb` | **Plantilla** clasificación binaria |
| `03-clasificacion-multiple.ipynb` | **Plantilla** clasificación multiclase |
| `01-regresion/` | Carpeta con **ejemplos** de regresión ya configurados |
| `02-clasificacion-binaria/` | Ejemplos binarios |
| `03-clasificacion-multiple/` | Ejemplos multiclase |

Los notebooks dentro de cada carpeta de ejemplos repiten el prefijo del tipo (`01-regresion-lineal-auto-mpg.ipynb`, etc.).

## Estructura del directorio

```
07.b-ejemplos-supervisados/
├── 01-regresion-lineal.ipynb              # plantilla
├── 02-clasificacion-binaria.ipynb
├── 03-clasificacion-multiple.ipynb
├── 01-regresion/                          # 3 ejemplos → ver README en la carpeta
├── 02-clasificacion-binaria/              # 3 ejemplos (breast, bank, thyroid)
├── 03-clasificacion-multiple/             # 3 ejemplos (iris, wine, thyroid 3 clases)
├── data/                                  # CSV (no versionados; ver download_datasets.sh)
└── download_datasets.sh
```

**12 notebooks** en total: 3 plantillas + 9 ejemplos.

## Inicio rápido

```bash
# Desde la raíz del repo
uv pip install -r requirements.txt

cd 07.b-ejemplos-supervisados
chmod +x download_datasets.sh
./download_datasets.sh
jupyter lab
```

- **Plantillas** (raíz): rutas `data/...`
- **Ejemplos** (subcarpetas): rutas `../data/...`

Para un CSV nuevo, copia la **plantilla** del tipo de problema y rellena exploración + CONFIG. Para ver un dataset resuelto, abre un notebook en la carpeta `01-regresion/`, `02-clasificacion-binaria/` o `03-clasificacion-multiple/`.

## Datasets tras `./download_datasets.sh`

| Archivo en `data/` | Notebook de ejemplo |
|--------------------|------------------------|
| `wine_quality_red.csv` | [01-regresion/01-regresion-lineal-wine-quality-red.ipynb](01-regresion/01-regresion-lineal-wine-quality-red.ipynb) |
| `auto_mpg.csv` | [01-regresion/01-regresion-lineal-auto-mpg.ipynb](01-regresion/01-regresion-lineal-auto-mpg.ipynb) |
| `diabetes.csv` | [01-regresion/01-regresion-lineal-diabetes.ipynb](01-regresion/01-regresion-lineal-diabetes.ipynb) |
| `breast_cancer.csv` | [02-clasificacion-binaria/02-clasificacion-binaria-breast-cancer.ipynb](02-clasificacion-binaria/02-clasificacion-binaria-breast-cancer.ipynb) |
| `bank_marketing.csv` | [02-clasificacion-binaria/02-clasificacion-binaria-bank-marketing.ipynb](02-clasificacion-binaria/02-clasificacion-binaria-bank-marketing.ipynb) |
| `thyroid.csv` | [02-clasificacion-binaria/02-clasificacion-binaria-thyroid.ipynb](02-clasificacion-binaria/02-clasificacion-binaria-thyroid.ipynb) |
| `iris.csv` | [03-clasificacion-multiple/03-clasificacion-multiple-iris.ipynb](03-clasificacion-multiple/03-clasificacion-multiple-iris.ipynb) |
| `wine_multiclass.csv` | [03-clasificacion-multiple/03-clasificacion-multiple-wine.ipynb](03-clasificacion-multiple/03-clasificacion-multiple-wine.ipynb) |
| `thyroid.csv` (3 clases) | [03-clasificacion-multiple/03-clasificacion-multiple-thyroid.ipynb](03-clasificacion-multiple/03-clasificacion-multiple-thyroid.ipynb) |

Detalle de URLs y alternativas: [data/README.md](data/README.md).

## Flujo de cada notebook

| Paso | Sección | Qué hace |
|------|---------|----------|
| 0 | Helpers | Imports, preprocesado, `split_train_val_test`; multiclase: `prepare_multiclass_target`. |
| 1 | Explorar CSV | Columnas, tipos, faltantes (antes de CONFIG). |
| 2 | CONFIG | Rutas, `RAW_LABEL_COL`, `CLASS_NAMES`, `DROP_COLS`; `build_models` (binario) o `build_models(n_classes)` (multiclase). |
| 3–5 | Carga y EDA | Codificar target 0/1 o 0..K-1; **multiclase:** `MODELS = build_models(N_CLASSES)` aquí. |
| 6 | Split | Train / val / test (estratificado en clasificación). |
| 7–8 | Preprocesado y benchmark | Entrena en train, compara en **val**. |
| 9 | Mejor modelo | Reentrena train+val, evalúa en **test**. |

## Plantillas (empieza aquí)

| Notebook | Problema |
|----------|----------|
| [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) | Target numérico continuo |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | Exactamente 2 clases |
| [03-clasificacion-multiple.ipynb](03-clasificacion-multiple.ipynb) | K clases (K ≥ 3; variable por dataset) |

## Carpetas de ejemplos

| Carpeta | Contenido |
|---------|-----------|
| [01-regresion/](01-regresion/) | Wine quality, Auto MPG, Diabetes |
| [02-clasificacion-binaria/](02-clasificacion-binaria/) | Breast cancer, Bank marketing |
| [03-clasificacion-multiple/](03-clasificacion-multiple/) | Iris, Wine cultivar, Thyroid (UCI) |

## CONFIG mínimo (plantilla)

```python
DATA_PATH = "data/mi_archivo.csv"   # en ejemplos: "../data/mi_archivo.csv"
CSV_SEP = ","
RAW_LABEL_COL = "columna_texto"     # None si el target ya es numérico
TARGET_COL = "target"               # 0/1 (binario) o 0..K-1 (multiclase)
CLASS_NAMES = None                  # None = inferir; o lista ordenada de K nombres
DROP_COLS = ["columna_texto", "id"]
TEST_SIZE = 0.2
VAL_SIZE = 0.25
```

## Cheat sheets relacionados

- [07.01 Fundamentos](../07-scikit-learn/07.01-scikit-learn-basics.ipynb)
- [07.02 Preprocesamiento](../07-scikit-learn/07.02-scikit-learn-preprocessing.ipynb)
- [07.03 Supervisados](../07-scikit-learn/07.03-scikit-learn-supervised-learning.ipynb)
- [07.05 Evaluación](../07-scikit-learn/07.05-scikit-learn-model-evaluation.ipynb)
- [07.06 Pipelines](../07-scikit-learn/07.06-scikit-learn-pipelines.ipynb)
