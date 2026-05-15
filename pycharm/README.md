# PyCharm — Redes neuronales (PyTorch)

Notebooks para ejecutar en **PyCharm** (o Jupyter): regresión y clasificación con **redes neuronales en PyTorch**. Sin Keras ni TensorFlow.

- **Carga de datos:** solo **pandas** y **NumPy** (leer CSV desde URL, normalizar, split).
- **Modelo y predicciones:** red neuronal con **PyTorch** (`nn.Module`, optimizador, entrenamiento).

## Contenido

| Notebook | Descripción |
|----------|-------------|
| [00-pycharm-cheat-sheet.ipynb](00-pycharm-cheat-sheet.ipynb) | **Cheat sheet PyCharm**: interfaz, atajos, navegación, edición, refactor, run/debug, Git, Jupyter, venvs, troubleshooting. |
| [04_training_dynamics.ipynb](04_training_dynamics.ipynb) | Dinámica de entrenamiento: **loss**, **optimizadores**, **regularización** y **schedulers** (con ejemplos 2D). |
| [01-regresion-lineal-red-neuronal.ipynb](01-regresion-lineal-red-neuronal.ipynb) | Regresión con red neuronal (dataset **Salary**). |
| [02-clasificacion-binaria.ipynb](02-clasificacion-binaria.ipynb) | Clasificación binaria (dataset **Pima Indians Diabetes**). |
| [03-clasificacion-multiple.ipynb](03-clasificacion-multiple.ipynb) | Clasificación en 3 clases (dataset **Wine** UCI). |

Los datasets se descargan desde la web al ejecutar cada notebook.

## Requisitos

- Python 3.8+
- `numpy`, `pandas`, `matplotlib`, **`torch`** (PyTorch)

Instalación: `pip install numpy pandas matplotlib torch`
