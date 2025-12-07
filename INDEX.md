# Índice de Cheat Sheets - Python, NumPy y Pandas

Este repositorio contiene cheat sheets organizados por tecnología en formato Jupyter Notebook (.ipynb).

## 📚 Python

### [python-basics.ipynb](python/python-basics.ipynb)
- Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
- Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

### [python-control-flow.ipynb](python/python-control-flow.ipynb)
- Condicionales (if/elif/else)
- Bucles (for, while)
- List Comprehensions
- Dict y Set Comprehensions
- zip() y enumerate()

### [python-functions-classes.ipynb](python/python-functions-classes.ipynb)
- Funciones (definición, argumentos variables, lambda)
- Clases y objetos
- Herencia
- Métodos especiales (dunder methods)
- Decoradores

### [python-modules-io.ipynb](python/python-modules-io.ipynb)
- Módulos y paquetes
- Manejo de archivos (I/O)
- JSON
- Manejo de excepciones

## 🔢 NumPy

### [numpy-basics.ipynb](numpy/numpy-basics.ipynb)
- Importar NumPy
- Creación de arrays
- Propiedades de arrays
- Tipos de datos (dtype)

### [numpy-operations.ipynb](numpy/numpy-operations.ipynb)
- Operaciones aritméticas
- Producto matricial
- Broadcasting
- Funciones de agregación

### [numpy-indexing-slicing.ipynb](numpy/numpy-indexing-slicing.ipynb)
- Indexación básica
- Fancy indexing (indexación avanzada)
- Modificación de arrays
- Concatenación y división

### [numpy-linear-algebra.ipynb](numpy/numpy-linear-algebra.ipynb)
- Álgebra lineal (determinante, inversa, autovalores, SVD, QR)
- Estadísticas avanzadas
- Generación de números aleatorios

## 🐼 Pandas

### [pandas-dataframes-series.ipynb](pandas/pandas-dataframes-series.ipynb)
- Crear Series
- Crear DataFrames
- Propiedades básicas

### [pandas-indexing-selection.ipynb](pandas/pandas-indexing-selection.ipynb)
- Selección de columnas
- Selección de filas
- Selección de filas y columnas (iloc, loc, at, iat)

### [pandas-data-manipulation.ipynb](pandas/pandas-data-manipulation.ipynb)
- Agregar y eliminar columnas
- Merge y Join
- GroupBy
- Pivot y Reshape

### [pandas-io-analysis.ipynb](pandas/pandas-io-analysis.ipynb)
- Lectura de archivos (CSV, Excel, JSON, Parquet, HTML)
- Escritura de archivos
- Análisis descriptivo
- Manejo de valores faltantes

## 🚀 Uso

Para usar estos notebooks, necesitas tener instalado:
- Python 3.8+
- Jupyter Notebook o JupyterLab
- NumPy
- Pandas

### Instalación rápida con uv

```bash
# Crear entorno virtual
uv venv

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Instalar dependencias
uv pip install jupyter numpy pandas
```

### Ejecutar Jupyter

```bash
jupyter notebook
# o
jupyter lab
```

## 📝 Notas

- Todos los notebooks están en español
- Los ejemplos son prácticos y listos para ejecutar
- Cada notebook está organizado por temas específicos para facilitar la consulta rápida

