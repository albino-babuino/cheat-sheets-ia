# 📚 Cheat Sheets - Python, NumPy, Pandas, Matplotlib y Scikit-learn

Repositorio completo de **cheat sheets** (hojas de referencia rápida) en formato Jupyter Notebook para **Python**, **NumPy**, **Pandas**, **Matplotlib** y **Scikit-learn**. Todos los notebooks están en español y organizados numéricamente para facilitar el aprendizaje progresivo.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 📖 Contenido

### 🔧 00 - Básicos (5 notebooks)

1. **[01-entornos-virtuales-uv.ipynb](00-basicos/01-entornos-virtuales-uv.ipynb)** - Entornos virtuales con uv
   - Instalación y configuración
   - Crear y gestionar entornos virtuales
   - Instalar paquetes con uv

2. **[02-gestion-paquetes-pip.ipynb](00-basicos/02-gestion-paquetes-pip.ipynb)** - Gestión de paquetes con pip
   - Instalar, actualizar y desinstalar paquetes
   - Gestionar requirements.txt
   - Cache y configuración

3. **[03-comandos-terminal-bash.ipynb](00-basicos/03-comandos-terminal-bash.ipynb)** - Comandos de terminal/bash
   - Navegación y gestión de archivos
   - Permisos y variables de entorno
   - Redirección y pipes

4. **[04-git-basico.ipynb](00-basicos/04-git-basico.ipynb)** - Git y GitHub
   - Control de versiones básico y avanzado
   - Ramas, merge, rebase y remotos
   - Stash, tags y resolución de conflictos
   - GitHub: Pull Requests, Issues, autenticación
   - Colaboración con forks y ramas compartidas
   - Mejores prácticas y resolución de problemas

5. **[05-markdown.ipynb](00-basicos/05-markdown.ipynb)** - Sintaxis Markdown
   - Formateo de texto
   - Listas, tablas y código
   - Enlaces e imágenes

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

### 📊 04 - Matplotlib (4 notebooks)

1. **[01-matplotlib-basics.ipynb](04-matplotlib/01-matplotlib-basics.ipynb)** - Fundamentos básicos
   - Importar Matplotlib
   - Primer gráfico básico
   - Agregar títulos y etiquetas
   - Múltiples líneas en un gráfico
   - Interface orientada a objetos (OO)
   - Guardar gráficos

2. **[02-matplotlib-customization.ipynb](04-matplotlib/02-matplotlib-customization.ipynb)** - Personalización
   - Colores (nombre, hexadecimal, RGB/RGBA)
   - Estilos de línea
   - Marcadores
   - Combinando estilos
   - Ancho de línea y transparencia
   - Personalizar ejes
   - Estilos predefinidos

3. **[03-matplotlib-plot-types.ipynb](04-matplotlib/03-matplotlib-plot-types.ipynb)** - Tipos de gráficos
   - Gráfico de barras (verticales y horizontales)
   - Gráfico de dispersión (Scatter)
   - Histogramas
   - Gráfico de área (Area Plot)
   - Gráfico de caja (Box Plot)
   - Gráfico de violín
   - Gráfico de pastel (Pie Chart)
   - Gráfico de barras agrupadas

4. **[04-matplotlib-advanced.ipynb](04-matplotlib/04-matplotlib-advanced.ipynb)** - Gráficos avanzados
   - Subplots
   - Subplots con diferentes tamaños (GridSpec)
   - Múltiples ejes (Twin Axes)
   - Gráficos 3D (superficie, línea, dispersión)
   - Anotaciones y texto
   - Líneas de referencia y regiones
   - Configuración global (rcParams)

### 🤖 05 - Scikit-learn (5 notebooks)

1. **[01-scikit-learn-basics.ipynb](05-scikit-learn/01-scikit-learn-basics.ipynb)** - Fundamentos básicos
   - Importar Scikit-learn
   - Datasets integrados (iris, wine, digits, diabetes, etc.)
   - Estructura básica de trabajo con modelos
   - Flujo completo: carga → división → entrenamiento → predicción → evaluación

2. **[02-scikit-learn-preprocessing.ipynb](05-scikit-learn/02-scikit-learn-preprocessing.ipynb)** - Preprocesamiento
   - Escalado de datos (StandardScaler, MinMaxScaler, RobustScaler, Normalizer)
   - Codificación de variables categóricas (LabelEncoder, OneHotEncoder, OrdinalEncoder)
   - Manejo de valores faltantes (SimpleImputer)
   - Transformaciones polinómicas (PolynomialFeatures)
   - Pipelines de preprocesamiento

3. **[03-scikit-learn-supervised-learning.ipynb](05-scikit-learn/03-scikit-learn-supervised-learning.ipynb)** - Aprendizaje supervisado
   - **Clasificación**: LogisticRegression, DecisionTree, RandomForest, SVM, KNN, Naive Bayes, Gradient Boosting
   - **Regresión**: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, SVR, KNN
   - Parámetros importantes y personalización de modelos

4. **[04-scikit-learn-unsupervised-learning.ipynb](05-scikit-learn/04-scikit-learn-unsupervised-learning.ipynb)** - Aprendizaje no supervisado
   - **Clustering**: K-Means, DBSCAN, Clustering Jerárquico Aglomerativo
   - **Reducción de dimensionalidad**: PCA, TruncatedSVD, NMF
   - t-SNE para visualización
   - Selección del número óptimo de componentes

5. **[05-scikit-learn-model-evaluation.ipynb](05-scikit-learn/05-scikit-learn-model-evaluation.ipynb)** - Evaluación de modelos
   - Métricas de clasificación (accuracy, precision, recall, F1, matriz de confusión)
   - Métricas de regresión (MSE, RMSE, MAE, R²)
   - Validación cruzada (K-Fold, Stratified K-Fold)
   - Búsqueda de hiperparámetros (GridSearchCV, RandomizedSearchCV)
   - Curvas ROC y AUC

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
uv pip install -r requirements.txt

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
├── 00-basicos/                      # Conocimientos básicos
│   ├── 01-entornos-virtuales-uv.ipynb
│   ├── 02-gestion-paquetes-pip.ipynb
│   ├── 03-comandos-terminal-bash.ipynb
│   ├── 04-git-basico.ipynb
│   └── 05-markdown.ipynb
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
├── 04-matplotlib/                   # Notebooks de Matplotlib
│   ├── 01-matplotlib-basics.ipynb
│   ├── 02-matplotlib-customization.ipynb
│   ├── 03-matplotlib-plot-types.ipynb
│   └── 04-matplotlib-advanced.ipynb
├── 05-scikit-learn/                 # Notebooks de Scikit-learn
│   ├── 01-scikit-learn-basics.ipynb
│   ├── 02-scikit-learn-preprocessing.ipynb
│   ├── 03-scikit-learn-supervised-learning.ipynb
│   ├── 04-scikit-learn-unsupervised-learning.ipynb
│   └── 05-scikit-learn-model-evaluation.ipynb
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

- ✅ **31 notebooks completos** con ejemplos prácticos
- ✅ **Todos los notebooks probados** y funcionando correctamente
- ✅ **Orden lógico de aprendizaje** con numeración
- ✅ **Documentación en español**
- ✅ **Ejemplos ejecutables** sin errores
- ✅ **Kernel configurado automáticamente**

## 🎯 Orden Recomendado de Aprendizaje

### Para Principiantes
1. Empieza con **00 - Básicos** (01-05) - Configuración y herramientas esenciales
2. Continúa con **01 - Python** (01-04)
3. Sigue con **02 - NumPy** (01-04)
4. Continúa con **03 - Pandas** (01-04)
5. Sigue con **04 - Matplotlib** (01-04) - Visualización de datos
6. Finaliza con **05 - Scikit-learn** (01-05) - Machine Learning

### Para Usuarios Intermedios
- Puedes saltar directamente a la sección que necesites
- Los notebooks están numerados para facilitar la navegación

## 📚 Recursos Adicionales

- [Documentación oficial de Python](https://docs.python.org/es/3/)
- [Documentación oficial de NumPy](https://numpy.org/doc/stable/)
- [Documentación oficial de Pandas](https://pandas.pydata.org/docs/)
- [Documentación oficial de Matplotlib](https://matplotlib.org/stable/)
- [Documentación oficial de Scikit-learn](https://scikit-learn.org/stable/)

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
