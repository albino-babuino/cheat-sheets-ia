# Datasets locales

Descarga los CSV por defecto desde la carpeta padre:

```bash
chmod +x download_datasets.sh
./download_datasets.sh
```

## Por notebook

| Notebook | Archivo | Target |
|----------|---------|--------|
| `01-regresion-lineal.ipynb` | *(el que pongas en `data/`)* | *(tu columna)* |
| `01-regresion/01-regresion-lineal-wine-quality-red.ipynb` | `wine_quality_red.csv` | `quality` |
| `01-regresion/01-regresion-lineal-auto-mpg.ipynb` | `auto_mpg.csv` | `mpg` |
| `01-regresion/01-regresion-lineal-diabetes.ipynb` | `diabetes.csv` | `disease_progression` |
| `02-clasificacion-binaria.ipynb` | *(tu CSV)* | *(2 clases)* |
| `02-clasificacion-binaria/02-clasificacion-binaria-breast-cancer.ipynb` | `breast_cancer.csv` | `target` (0/1) |
| `02-clasificacion-binaria/02-clasificacion-binaria-bank-marketing.ipynb` | `bank_marketing.csv` | `y` (yes/no) |
| `03-clasificacion-multiple.ipynb` | *(tu CSV)* | *(3+ clases)* |
| `03-clasificacion-multiple/03-clasificacion-multiple-iris.ipynb` | `iris.csv` | `species` |
| `03-clasificacion-multiple/03-clasificacion-multiple-wine.ipynb` | `wine_multiclass.csv` | `target` (0/1/2) |
| `03-clasificacion-multiple/03-clasificacion-multiple-thyroid.ipynb` | `thyroid.csv` | `class_label` (3 clases) |

Rutas en código: `data/...` en plantillas (raíz), `../data/...` en ejemplos (subcarpetas).

## URLs de descarga manual

- **Wine Quality (red):** https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
- **Iris:** https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
- **Breast Cancer:** generado con `sklearn.datasets.load_breast_cancer` en `download_datasets.sh`
- **Bank Marketing:** https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv
- **Auto MPG:** https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data
- **Thyroid (UCI allbp):** https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/allbp.data
- **Diabetes:** archivo local `diabetes.csv`
