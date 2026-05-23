# Esquemas sklearn + PyTorch (13)

Notebooks **MVP** con datos tabulares en CSV: mismo esquema manual que un proyecto supervisado completo, añadiendo un **MLP en PyTorch** y una **comparación final** sklearn + red.

## Flujo común (todos los notebooks)

| Paso | Contenido |
|------|-----------|
| 1–4 | CSV en [`data/`](data/), target, features, tratamiento manual |
| 5 | Split **train / val / test** (`split_train_val_test`, ~60 % / 20 % / 20 %) |
| 6 | **Entrenar sklearn**: `build_sklearn_pipeline()` → `transformacion` + `estandarizado` + `modelo` |
| 7 | **Entrenar PyTorch**: MLP + Adam (imputación y escalado separados, solo stats de train) |
| 8 | **Análisis comparativo**: métricas en val/test, tabla única, ganador por **val**, reporte en **test** |

Los pasos **6–7** solo entrenan. Las predicciones y la tabla comparativa están **solo en el paso 8**.

> **Limitación didáctica:** imputación y dummies en pandas (pasos 3–4) antes del split; en el `Pipeline` sklearn van **por separado** `transformacion` (`SimpleImputer`) y `estandarizado` (`StandardScaler`). En [07.b](../07.b-ejemplos-supervisados/) también el one-hot va dentro del `Pipeline` ajustado solo en train.

## Notebooks

| Archivo | Target | sklearn (paso 6) | PyTorch (paso 7) |
|---------|--------|------------------|------------------|
| [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) | `Precio` | 10 regresores | `HousePriceNet` + `MSELoss` |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | 0/1 | 12 clasificadores | `TabularBinaryNet` + `BCEWithLogitsLoss` |
| [03-clasificacion-multiclase.ipynb](03-clasificacion-multiclase.ipynb) | 0..K-1 | `build_models(N_CLASSES)` | `TabularMultiNet` + `CrossEntropyLoss` |

## Requisitos y kernel de Jupyter

Desde la **raíz del repo** (no uses el Python del sistema: ahí no está `scikit-learn`).

```bash
cd cheat-sheets-ia
uv venv
uv pip install -r requirements.txt
uv run python -m ipykernel install --user --name cheat-sheets-ia --display-name "cheat-sheets-ia (.venv)"
```

En Cursor/VS Code: **Select Kernel** → `cheat-sheets-ia (.venv)` (o el intérprete `.venv/bin/python`).

Comprueba en una celda:

```python
import sys
from sklearn.impute import SimpleImputer
print(sys.executable)  # debe terminar en .../cheat-sheets-ia/.venv/bin/python
```

| Error | Causa habitual |
|-------|----------------|
| `No module named 'sklearn'` | Kernel = Python del sistema |
| `No module named 'sklearn.impute'` | Paquete `sklearn` de PyPI en vez de `scikit-learn`, o sklearn muy viejo |

Comenta entradas en `build_models()` si quieres acortar la ejecución (p. ej. SVC u OvO/OvR).

## Material relacionado

| Tema | Carpeta |
|------|---------|
| Esquema manual solo sklearn | [07.a](../07.a-esquemas-supervisados/) |
| Pipelines completos, CV, ColumnTransformer | [07.b](../07.b-ejemplos-supervisados/) |
| PyTorch (datasets URL, más teoría) | [12-pytorch](../12-pytorch/) |
| Cheat sheet PyTorch | [12.00](../12-pytorch/00-pytorch-cheat-sheet.ipynb) |
| Pandas ↔ PyTorch | [04.06](../04-pandas/04.06-numpy-pandas-pytorch-interop.ipynb) |
