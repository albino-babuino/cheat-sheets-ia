# Ejemplos no supervisados (Scikit-learn)

Notebooks **completos** de **clustering** y **reducción de dimensionalidad** que aplican [07.04 Aprendizaje no supervisado](../07-scikit-learn/07.04-scikit-learn-unsupervised-learning.ipynb) y [07.06 Pipelines](../07-scikit-learn/07.06-scikit-learn-pipelines.ipynb): CSV local, EDA, `ColumnTransformer`, **`Pipeline(preprocess → PCA)`** y **`Pipeline(preprocess → KMeans)`**.

No hay variable objetivo en el entrenamiento. Si el CSV trae etiquetas (`LABEL_COL`), solo se usan para **validación externa** (p. ej. ARI), nunca dentro del `fit`.

## Estructura

```
07.c-ejemplos-no-supervisados/
├── 01-clustering-kmeans-pca.ipynb     # plantilla
├── 01-clustering/
│   ├── 01-kmeans-pca-iris.ipynb
│   └── 01-kmeans-pca-wine-quality.ipynb
├── data/
└── download_datasets.sh
```

**3 notebooks:** 1 plantilla + 2 ejemplos.

## Inicio rápido

```bash
# Desde la raíz del repo
uv pip install -r requirements.txt

cd 07.c-ejemplos-no-supervisados
chmod +x download_datasets.sh
./download_datasets.sh
jupyter lab
```

- **Plantilla** (raíz): rutas `data/...`
- **Ejemplos** (`01-clustering/`): rutas `../data/...`

También puedes copiar `iris.csv` y `wine_quality_red.csv` desde [07.b-ejemplos-supervisados/data/](../07.b-ejemplos-supervisados/data/).

## Datasets

| Archivo en `data/` | Notebook |
|--------------------|----------|
| `iris.csv` | [01-kmeans-pca-iris.ipynb](01-clustering/01-kmeans-pca-iris.ipynb) |
| `wine_quality_red.csv` | [01-kmeans-pca-wine-quality.ipynb](01-clustering/01-kmeans-pca-wine-quality.ipynb) |

## Flujo de cada notebook

| Paso | Sección | Qué hace |
|------|---------|----------|
| 0 | Helpers | `build_preprocess`, `choose_k_elbow`, métricas silhouette / ARI. |
| 1 | Explorar CSV | Tipos y faltantes. |
| 2 | CONFIG | `LABEL_COL` (opcional), `N_CLUSTERS`, `K_RANGE`, `N_COMPONENTS_PCA`. |
| 3–4 | Carga y EDA | Lectura y resumen. |
| 5 | Features | Matriz **X** (sin `LABEL_COL`). |
| 6 | Preprocesado | `ColumnTransformer` compartido. |
| 7 | PCA | `Pipeline` → varianza explicada, scatter PC1–PC2. |
| 8 | K-Means | Codo + silhouette; `Pipeline(preprocess → KMeans)`. |
| 9 | Resultados | Clusters, scatter, crosstab opcional vs `LABEL_COL`. |

## CONFIG mínimo (plantilla)

```python
DATA_PATH = "data/mi_archivo.csv"
CSV_SEP = ","
LABEL_COL = None          # solo validación; no entra en X
DROP_COLS = ["id"]
N_CLUSTERS = 3            # None → elegir K por silhouette
K_RANGE = range(2, 11)
N_COMPONENTS_PCA = 2
RANDOM_STATE = 42
```

## Cheat sheets relacionados

- [07.04 No supervisado](../07-scikit-learn/07.04-scikit-learn-unsupervised-learning.ipynb)
- [07.06 Pipelines](../07-scikit-learn/07.06-scikit-learn-pipelines.ipynb)
- [07.b Ejemplos supervisados](../07.b-ejemplos-supervisados/)
