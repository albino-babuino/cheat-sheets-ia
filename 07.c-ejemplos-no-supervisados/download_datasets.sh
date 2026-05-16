#!/usr/bin/env bash
# Descarga CSV para 07.c-ejemplos-no-supervisados (iris, wine quality red).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data

if command -v uv >/dev/null 2>&1; then
  PYTHON="uv run python"
else
  PYTHON="python3"
fi

echo "==> Iris"
curl -fsSL -o data/iris_raw.data \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
$PYTHON << 'PY'
import pandas as pd

cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
df = pd.read_csv("data/iris_raw.data", names=cols)
df = df.dropna(how="all")
df.to_csv("data/iris.csv", index=False)
print(f"  iris.csv: {len(df)} filas")
PY

echo "==> Wine Quality (red)"
curl -fsSL -o data/wine_quality_red.csv \
  "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

echo "Listo. CSV en data/"
