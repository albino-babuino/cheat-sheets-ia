# 📚 Cheat Sheets - Python, NumPy y Pandas

Repositorio completo de **cheat sheets** (hojas de referencia rápida) en formato Jupyter Notebook para **Python**, **NumPy** y **Pandas**. Todos los notebooks están en español y organizados numéricamente para facilitar el aprendizaje progresivo.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 📖 Contenido

### 🐍 01 - Python (4 notebooks)

1. **[01-python-basics.ipynb](01-python/01-python-basics.ipynb)** - Fundamentos básicos
   - Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
   - Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

2. **[02-python-control-flow.ipynb](01-python/02-python-control-flow.ipynb)** - Control de flujo
   - Condicionales (if/elif/else)
   - Bucles (for, while)
   - List, Dict y Set Comprehensions
   - zip() y enumerate()

3. **[03-python-functions-classes.ipynb](01-python/03-python-functions-classes.ipynb)** - Funciones y clases
   - Funciones (definición, argumentos variables, lambda)
   - Clases y objetos
   - Herencia
   - Métodos especiales (dunder methods)
   - Decoradores

4. **[04-python-modules-io.ipynb](01-python/04-python-modules-io.ipynb)** - Módulos e I/O
   - Módulos y paquetes
   - Manejo de archivos (I/O)
   - JSON
   - Manejo de excepciones

### 🔢 02 - NumPy (4 notebooks)

1. **[01-numpy-basics.ipynb](02-numpy/01-numpy-basics.ipynb)** - Fundamentos básicos
   - Importar NumPy
   - Creación de arrays
   - Propiedades de arrays
   - Tipos de datos (dtype)

2. **[02-numpy-operations.ipynb](02-numpy/02-numpy-operations.ipynb)** - Operaciones
   - Operaciones aritméticas
   - Producto matricial
   - Broadcasting
   - Funciones de agregación

3. **[03-numpy-indexing-slicing.ipynb](02-numpy/03-numpy-indexing-slicing.ipynb)** - Indexación y slicing
   - Indexación básica
   - Fancy indexing (indexación avanzada)
   - Modificación de arrays
   - Concatenación y división

4. **[04-numpy-linear-algebra.ipynb](02-numpy/04-numpy-linear-algebra.ipynb)** - Álgebra lineal y estadísticas
   - Álgebra lineal (determinante, inversa, autovalores, SVD, QR)
   - Estadísticas avanzadas
   - Generación de números aleatorios

### 🐼 03 - Pandas (4 notebooks)

1. **[01-pandas-dataframes-series.ipynb](03-pandas/01-pandas-dataframes-series.ipynb)** - Series y DataFrames
   - Crear Series
   - Crear DataFrames
   - Propiedades básicas

2. **[02-pandas-indexing-selection.ipynb](03-pandas/02-pandas-indexing-selection.ipynb)** - Indexación y selección
   - Selección de columnas
   - Selección de filas
   - Selección de filas y columnas (iloc, loc, at, iat)

3. **[03-pandas-data-manipulation.ipynb](03-pandas/03-pandas-data-manipulation.ipynb)** - Manipulación de datos
   - Agregar y eliminar columnas
   - Merge y Join
   - GroupBy
   - Pivot y Reshape

4. **[04-pandas-io-analysis.ipynb](03-pandas/04-pandas-io-analysis.ipynb)** - I/O y análisis
   - Lectura de archivos (CSV, Excel, JSON, Parquet, HTML)
   - Escritura de archivos
   - Análisis descriptivo
   - Manejo de valores faltantes

## 🚀 Inicio Rápido

### Requisitos Previos

- Python 3.8 o superior
- `uv` (gestor de paquetes rápido) o `pip` tradicional

### Instalación

1. **Clonar el repositorio:**
```bash
git clone git@github.com:albino-babuino/cheat-sheets-ia.git
cd cheat-sheets-ia
```

2. **Crear y activar el entorno virtual:**
```bash
# Con uv (recomendado)
uv venv
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# O con venv tradicional
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
```

3. **Instalar dependencias:**
```bash
# Con uv
uv pip install jupyter numpy pandas

# O con pip
pip install -r requirements.txt
```

4. **Iniciar Jupyter:**
```bash
jupyter notebook
# o
jupyter lab
```

## 📁 Estructura del Proyecto

```
cheat-sheets-ia/
├── 01-python/                       # Notebooks de Python
│   ├── 01-python-basics.ipynb
│   ├── 02-python-control-flow.ipynb
│   ├── 03-python-functions-classes.ipynb
│   └── 04-python-modules-io.ipynb
├── 02-numpy/                        # Notebooks de NumPy
│   ├── 01-numpy-basics.ipynb
│   ├── 02-numpy-operations.ipynb
│   ├── 03-numpy-indexing-slicing.ipynb
│   └── 04-numpy-linear-algebra.ipynb
├── 03-pandas/                       # Notebooks de Pandas
│   ├── 01-pandas-dataframes-series.ipynb
│   ├── 02-pandas-indexing-selection.ipynb
│   ├── 03-pandas-data-manipulation.ipynb
│   └── 04-pandas-io-analysis.ipynb
├── .venv/                           # Entorno virtual
├── INDEX.md                         # Índice detallado
├── README.md                        # Este archivo
└── requirements.txt                 # Dependencias
```

## 🔧 Configuración del Kernel de Jupyter

El kernel de Jupyter está configurado automáticamente para usar el entorno virtual del proyecto. Todos los notebooks están configurados para usar el kernel `python3` que apunta a `.venv/bin/python`.

**Ubicación del kernel:** `.venv/share/jupyter/kernels/python3/`

No necesitas configurar nada manualmente. Al abrir cualquier notebook, se usará automáticamente el entorno virtual correcto.

## 📝 Características

- ✅ **12 notebooks completos** con ejemplos prácticos
- ✅ **Todos los notebooks probados** y funcionando correctamente
- ✅ **Orden lógico de aprendizaje** con numeración
- ✅ **Documentación en español**
- ✅ **Ejemplos ejecutables** sin errores
- ✅ **Kernel configurado automáticamente**

## 🎯 Orden Recomendado de Aprendizaje

### Para Principiantes
1. Empieza con **01 - Python** (01-04)
2. Continúa con **02 - NumPy** (01-04)
3. Finaliza con **03 - Pandas** (01-04)

### Para Usuarios Intermedios
- Puedes saltar directamente a la sección que necesites
- Los notebooks están numerados para facilitar la navegación

## 📚 Recursos Adicionales

- [Documentación oficial de Python](https://docs.python.org/es/3/)
- [Documentación oficial de NumPy](https://numpy.org/doc/stable/)
- [Documentación oficial de Pandas](https://pandas.pydata.org/docs/)

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Si encuentras algún error o quieres añadir contenido, por favor:

1. Abre un issue describiendo el problema o mejora
2. O crea un pull request con tus cambios

## 📄 Licencia

Este repositorio contiene material educativo de referencia rápida. Siéntete libre de usarlo para aprender y compartir conocimiento.

## ⭐ Estrellas

Si este repositorio te resulta útil, ¡considera darle una estrella! ⭐

---

**Creado con ❤️ para la comunidad de Python, Data Science y Machine Learning**
