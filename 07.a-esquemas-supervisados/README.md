# Esquemas supervisados (07.a)

Notebooks **MVP (mínima viable)**: CSV → pandas → **tratamiento manual** → split **train / val / test** → entrenar varios modelos sklearn → **análisis comparativo** en un apartado final.

## Flujo común

| Fase | Qué hace |
|------|----------|
| Datos | CSV en [`data/`](data/), tipos, faltantes, codificación del target |
| Split | `split_train_val_test` (`TEST_SIZE=0.2`, `VAL_SIZE=0.25` → ~60 % / 20 % / 20 %) |
| Entrenar | `build_models()` + bucle: `fit` en **train** → dict `pipelines` |
| Análisis | `predict` en **val** y **test**, tabla de métricas, **ganador por val**, reporte en test |

> **Limitación didáctica:** imputación y dummies se calculan sobre todo el `df` antes del split. En [07.b](../07.b-ejemplos-supervisados/) el preprocesado va **dentro** del `Pipeline` ajustado solo en train.

## Tratamiento manual por tipo

| Tipo | Dónde | Tratamiento mínimo |
|------|-------|-------------------|
| Target **numérico** | Regresión (`Precio`) | `dropna` en target |
| Target **categórico** binario | Binaria (`Etiqueta_Texto`) | `map` → 0/1 |
| Target **categórico** K clases | Multiclase (`especie`) | `map` → 0..K-1 |
| Feature **numérica** con NaN | Los tres | `fillna(mediana)` |
| Feature **categórica** | Regresión (`Zona`) | `fillna` + `pd.get_dummies` |

Los CSV incluyen **faltantes a propósito** para practicar.

## Notebooks

| Archivo | Pasos finales | Target | Modelos |
|---------|---------------|--------|---------|
| [01-regresion-lineal.ipynb](01-regresion-lineal.ipynb) | 6 entrenar · 7 análisis | `Precio` | 10 regresores (`build_models`) |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | 5 entrenar · 6 análisis | `no`/`si` → 0/1 | 12 clasificadores |
| [03-clasificacion-multiclase.ipynb](03-clasificacion-multiclase.ipynb) | 5 entrenar · 6 análisis | `especie` → 0..K-1 | Igual (`build_models(N_CLASSES)`) |

Lista de modelos alineada con [07.b](../07.b-ejemplos-supervisados/): lineales, árboles, boosting, KNN, XGBoost, CatBoost (clasificación: también SVC, OvO, OvR).

## Requisitos

```bash
pip install numpy pandas scikit-learn xgboost catboost
```

Comenta entradas en `build_models()` para acortar ejecuciones (p. ej. SVC u OvO/OvR).

## Orden del curso 07

1. **07-scikit-learn/** — teoría  
2. **07.a** (aquí) — esquema manual + pipeline simple  
3. **[13-esquemas-sklearn-pytorch](../13-esquemas-sklearn-pytorch/)** — sklearn + MLP PyTorch  
4. **07.b** — `ColumnTransformer`, imputer, one-hot en pipeline, CV  
5. **07.c** — no supervisado  
