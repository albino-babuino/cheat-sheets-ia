#!/usr/bin/env bash
# Descarga los CSV por defecto para los notebooks de 07.b-ejemplos-supervisados.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

echo "==> Wine Quality (red)"
curl -fsSL -o data/wine_quality_red.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

echo "==> Iris"
curl -fsSL -o data/iris_raw.data \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
if command -v uv >/dev/null 2>&1; then
  PYTHON="uv run python"
else
  PYTHON="python3"
fi

$PYTHON << 'PY'
import pandas as pd
cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("data/iris_raw.data", names=cols)
df = df.dropna(how="all")
df.to_csv("data/iris.csv", index=False)
print(f"  iris.csv: {len(df)} filas")
PY

echo "==> Auto MPG (UCI)"
curl -fsSL -o data/auto_mpg_raw.data \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
$PYTHON << 'PY'
import pandas as pd

cols = [
    "mpg", "cylinders", "displacement", "horsepower", "weight",
    "acceleration", "model_year", "origin", "car_name",
]
rows = []
with open("data/auto_mpg_raw.data", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if '"' in line:
            i, j = line.index('"'), line.rindex('"')
            head = line[:i].split()
            car = line[i + 1 : j]
            rows.append(head + [car])
        else:
            parts = line.split()
            rows.append(parts[:8] + [" ".join(parts[8:])])

df = pd.DataFrame(rows, columns=cols)
for c in cols[:-1]:
    df[c] = pd.to_numeric(df[c].replace("?", pd.NA), errors="coerce")
df.to_csv("data/auto_mpg.csv", index=False)
print(f"  auto_mpg.csv: {len(df)} filas, target=mpg, nulos={int(df.isna().sum().sum())}")
PY

echo "==> Breast Cancer (Wisconsin, vía sklearn)"
$PYTHON << 'PY'
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["diagnosis"] = df["target"].map({0: "B", 1: "M"})
df.to_csv("data/breast_cancer.csv", index=False)
print(f"  breast_cancer.csv: {len(df)} filas")
PY

echo "==> Bank Marketing (UCI)"
curl -fsSL -o data/bank_marketing.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional-full.csv"
$PYTHON << 'PY'
import pandas as pd
df = pd.read_csv("data/bank_marketing.csv", sep=";")
print(f"  bank_marketing.csv: {len(df)} filas, target=y ({df['y'].unique().tolist()})")
PY

echo "==> Wine multiclass (cultivar, vía sklearn)"
$PYTHON << 'PY'
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["cultivar"] = df["target"].map(dict(enumerate(data.target_names)))
df.to_csv("data/wine_multiclass.csv", index=False)
print(f"  wine_multiclass.csv: {len(df)} filas, {df['target'].nunique()} clases")
PY

echo "Listo. Archivos en data/"
