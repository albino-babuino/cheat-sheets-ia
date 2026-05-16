# 12 - PyTorch — Redes neuronales

Notebooks de regresión y clasificación con **redes neuronales en PyTorch**. Sin Keras ni TensorFlow.

- **Carga de datos:** **pandas** y **NumPy** (leer CSV desde URL, normalizar, split).
- **Modelo y entrenamiento:** **PyTorch** (`nn.Module`, optimizador, bucle de entrenamiento).

## Contenido

| Notebook | Descripción |
|----------|-------------|
| [00-pytorch-cheat-sheet.ipynb](00-pytorch-cheat-sheet.ipynb) | Referencia rápida: tensores, `autograd`, capas, pérdidas, optimizadores y bucle de entrenamiento. |
| [01-regresion-lineal-red-neuronal.ipynb](01-regresion-lineal-red-neuronal.ipynb) | Regresión con red neuronal (dataset **Salary**). |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | Clasificación binaria (dataset **Pima Indians Diabetes**). |
| [03-clasificacion-multiple.ipynb](03-clasificacion-multiple.ipynb) | Clasificación en 3 clases (dataset **Wine** UCI). |

Los datasets se descargan desde la web al ejecutar cada notebook.

**Conversiones con pandas/NumPy:** ver [04.06-numpy-pandas-pytorch-interop.ipynb](../04-pandas/04.06-numpy-pandas-pytorch-interop.ipynb).

## Requisitos

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, **`torch`** (PyTorch)

Instalación: `pip install numpy pandas matplotlib torch`
