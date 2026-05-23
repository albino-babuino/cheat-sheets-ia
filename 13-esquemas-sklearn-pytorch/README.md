# Esquemas sklearn + PyTorch (13)

Notebooks **MVP** de aprendizaje supervisado con datos tabulares en CSV:

1. CSV → pandas → **tratamiento manual** (tipos, faltantes, codificación del target).
2. Split **train / val / test** (`split_train_val_test`, ~60 % / 20 % / 20 %).
3. Varios modelos **sklearn** en `make_pipeline(StandardScaler, …)` (`build_models()`).
4. **Red neuronal mínima** en PyTorch (MLP + Adam; entrena solo en train).
5. **Análisis comparativo** (paso final aparte): `predict` en val/test, métricas, tabla sklearn + PyTorch; elige el mejor en **val** y reporta en **test**.

Los CSV de práctica están en [`data/`](data/) (incluyen faltantes a propósito).

## Notebooks

| Archivo | Target | sklearn (paso 6) | PyTorch (paso 7) |
|---------|--------|------------------|------------------|
| [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) | `Precio` | 10 regresores (linear, boosting, XGBoost, CatBoost, …) | `HousePriceNet` |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | 0/1 | 12 clasificadores | `TabularBinaryNet` + `BCEWithLogitsLoss` |
| [03-clasificacion-multiclase.ipynb](03-clasificacion-multiclase.ipynb) | 0..K-1 | Igual con `build_models(N_CLASSES)` | `TabularMultiNet` + `CrossEntropyLoss` |

El **paso 8** muestra una única tabla con todos los modelos y elige el mejor en test.

> **Limitación didáctica:** imputación y dummies se calculan sobre todo el `df` antes del split. En proyectos reales conviene encapsularlo en un `Pipeline` ajustado solo en train.

## Requisitos

```bash
pip install numpy pandas scikit-learn torch xgboost catboost
```

Comenta entradas en `build_models()` si quieres acortar la ejecución (p. ej. SVC u OvO/OvR).

## Material relacionado

| Tema | Carpeta |
|------|---------|
| PyTorch (datasets URL, más teoría) | [12-pytorch](../12-pytorch/) |
| Pipelines completos, CV, ColumnTransformer | [07.b-ejemplos-supervisados](../07.b-ejemplos-supervisados/) |
| Cheat sheet PyTorch | [12.00](../12-pytorch/00-pytorch-cheat-sheet.ipynb) |
| Pandas ↔ PyTorch | [04.06](../04-pandas/04.06-numpy-pandas-pytorch-interop.ipynb) |
