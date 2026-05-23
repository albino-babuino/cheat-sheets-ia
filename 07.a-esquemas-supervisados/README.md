# Esquemas supervisados (07.a)

Notebooks **MVP (mínima viable)**: CSV → pandas → **tratamiento manual** de tipos y faltantes → split → `Pipeline` (solo escalado + modelo).

## Qué cubre el tratamiento manual

| Tipo | Dónde | Tratamiento mínimo |
|------|-------|-------------------|
| Target **numérico** | Regresión (`Precio`) | `dropna` en target |
| Target **categórico** binario | Binaria (`Etiqueta_Texto`) | `map` → 0/1 |
| Target **categórico** K clases | Multiclase (`especie`) | `map` → 0..K-1 |
| Feature **numérica** con NaN | Los tres | `fillna(mediana)` |
| Feature **categórica** | Regresión (`Zona`) | `fillna` + `pd.get_dummies` |

Los CSV en [`data/`](data/) incluyen **valores faltantes a propósito** para practicar.

> **Limitación didáctica:** las medianas/dummies se calculan sobre todo el `df` antes del split. En [07.b](../07.b-ejemplos-supervisados/) eso va dentro del `Pipeline` ajustado **solo en train** (evita leakage).

## Notebooks

Cada uno termina con un **bucle** sobre `MODELS`: mismo `make_pipeline(StandardScaler(), modelo)`, tabla comparativa en test.

| Archivo | Target | Modelos (`build_models`, igual que [07.b](../07.b-ejemplos-supervisados/)) |
|---------|--------|---------------------|
| [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) | `Precio` | Linear, Ridge, Lasso, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, KNN, XGBoost, CatBoost |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | `no`/`si` → 0/1 | Logistic, SGD, SVC, OvO, OvR, KNN, DecisionTree, RandomForest, GradientBoosting, HistGradientBoosting, XGBoost, CatBoost |
| [03-clasificacion-multiclase.ipynb](03-clasificacion-multiclase.ipynb) | `especie` → 0,1,2 | Misma lista que binaria (`build_models(N_CLASSES)`) |

Requiere `xgboost` y `catboost` instalados (como en 07.b). Comenta entradas en `build_models()` para acortar ejecuciones.

## Orden del curso 07

1. **07-scikit-learn/** — teoría  
2. **07.a** (aquí) — esquema manual + pipeline simple  
3. **[13-esquemas-sklearn-pytorch](../13-esquemas-sklearn-pytorch/)** — lo mismo + MLP PyTorch al final  
4. **07.b** — `ColumnTransformer`, imputer, one-hot en pipeline, benchmark  
5. **07.c** — no supervisado  
