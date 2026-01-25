# 📚 Cheat Sheets - Python, Data Science, Machine Learning e IA Moderna

Repositorio completo de **cheat sheets** (hojas de referencia rápida) en formato Jupyter Notebook para **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Scikit-learn**, **Algoritmos Clásicos de ML**, **Estadística** e **IA Moderna** (Deep Learning, Transformers, CNN, RNN/LSTM). Todos los notebooks están en español y organizados numéricamente para facilitar el aprendizaje progresivo.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

## 📖 Contenido

### 🔧 01 - Básicos (5 notebooks)

1. **[01.01-entornos-virtuales-uv.ipynb](01-basicos/01.01-entornos-virtuales-uv.ipynb)** - Entornos virtuales con uv
   - Instalación y configuración
   - Crear y gestionar entornos virtuales
   - Instalar paquetes con uv

2. **[01.02-gestion-paquetes-pip.ipynb](01-basicos/01.02-gestion-paquetes-pip.ipynb)** - Gestión de paquetes con pip
   - Instalar, actualizar y desinstalar paquetes
   - Gestionar requirements.txt
   - Cache y configuración

3. **[01.03-comandos-terminal-bash.ipynb](01-basicos/01.03-comandos-terminal-bash.ipynb)** - Comandos de terminal/bash
   - Navegación y gestión de archivos
   - Permisos y variables de entorno
   - Redirección y pipes

4. **[01.04-git-basico.ipynb](01-basicos/01.04-git-basico.ipynb)** - Git y GitHub
   - Control de versiones básico y avanzado
   - Ramas, merge, rebase y remotos
   - Stash, tags y resolución de conflictos
   - GitHub: Pull Requests, Issues, autenticación
   - Colaboración con forks y ramas compartidas
   - Mejores prácticas y resolución de problemas

5. **[01.05-markdown.ipynb](01-basicos/01.05-markdown.ipynb)** - Sintaxis Markdown
   - Formateo de texto
   - Listas, tablas y código
   - Enlaces e imágenes

### 🐍 02 - Python (5 notebooks)

1. **[02.01-python-basics.ipynb](02-python/02.01-python-basics.ipynb)** - Fundamentos básicos
   - Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
   - Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

2. **[02.02-python-control-flow.ipynb](02-python/02.02-python-control-flow.ipynb)** - Control de flujo
   - Condicionales (if/elif/else)
   - Bucles (for, while)
   - List, Dict y Set Comprehensions
   - zip() y enumerate()

3. **[02.03-python-functions.ipynb](02-python/02.03-python-functions.ipynb)** - Funciones
   - Definición básica de funciones
   - Argumentos variables (*args, **kwargs)
   - Funciones lambda (anónimas)
   - Decoradores
   - Funciones como objetos de primera clase

4. **[02.04-python-classes-oop.ipynb](02-python/02.04-python-classes-oop.ipynb)** - Programación Orientada a Objetos (POO)
   - Clases y objetos
   - Atributos de clase vs instancia
   - Métodos de clase y estáticos
   - Herencia (simple y múltiple)
   - Métodos especiales (dunder methods)
   - Propiedades (getters y setters)
   - Encapsulación

5. **[02.05-python-modules-io.ipynb](02-python/02.05-python-modules-io.ipynb)** - Módulos e I/O
   - Módulos y paquetes
   - Manejo de archivos (I/O)
   - JSON
   - Manejo de excepciones

### 🔢 03 - NumPy (4 notebooks)

1. **[03.01-numpy-basics.ipynb](03-numpy/03.01-numpy-basics.ipynb)** - Fundamentos básicos
   - Importar NumPy
   - Creación de arrays
   - Propiedades de arrays
   - Tipos de datos (dtype)
   - I/O (Entrada/Salida): guardar y cargar arrays (np.save, np.load, np.savetxt, np.loadtxt)

2. **[03.02-numpy-operations.ipynb](03-numpy/03.02-numpy-operations.ipynb)** - Operaciones
   - Operaciones aritméticas
   - Producto matricial
   - Broadcasting
   - Funciones de agregación

3. **[03.03-numpy-indexing-slicing.ipynb](03-numpy/03.03-numpy-indexing-slicing.ipynb)** - Indexación y slicing
   - Indexación básica
   - Fancy indexing (indexación avanzada)
   - Comparaciones elemento a elemento
   - Copia vs vista (views vs copies)
   - Manipulación de arrays (transpose, reshape, append, insert, delete)
   - Ordenamiento de arrays

4. **[03.04-numpy-linear-algebra.ipynb](03-numpy/03.04-numpy-linear-algebra.ipynb)** - Álgebra lineal y estadísticas
   - Álgebra lineal (determinante, inversa, autovalores, SVD, QR)
   - Estadísticas avanzadas
   - Generación de números aleatorios

### 🐼 04 - Pandas (5 notebooks)

1. **[04.01-pandas-dataframes-series.ipynb](04-pandas/04.01-pandas-dataframes-series.ipynb)** - Series y DataFrames
   - Crear Series
   - Crear DataFrames
   - Propiedades básicas

2. **[04.02-pandas-indexing-selection.ipynb](04-pandas/04.02-pandas-indexing-selection.ipynb)** - Indexación y selección
   - Selección de columnas
   - Selección de filas
   - Selección de filas y columnas (iloc, loc, at, iat)

3. **[04.03-pandas-data-manipulation.ipynb](04-pandas/04.03-pandas-data-manipulation.ipynb)** - Manipulación de datos
   - Agregar y eliminar columnas/filas
   - Manipulación de índice (reindex, set_index, reset_index, sort_index)
   - Ordenamiento de valores
   - Merge y Join
   - GroupBy (agregación, transformación, filtrado)
   - Pivot y Reshape (pivot_table, melt, stack, unstack)

4. **[04.04-pandas-io-analysis.ipynb](04-pandas/04.04-pandas-io-analysis.ipynb)** - I/O y análisis
   - Lectura de archivos (CSV, Excel, JSON, Parquet, HTML)
   - Escritura de archivos
   - Análisis descriptivo
   - Manejo de valores faltantes

5. **[04.05-pandas-visualization.ipynb](04-pandas/04.05-pandas-visualization.ipynb)** - Visualización de datos
   - Gráficos de línea, barras, histogramas
   - Box plots, scatter plots, pie charts
   - Gráficos de área y hexbin
   - KDE (estimación de densidad)
   - Personalización de gráficos
   - Subplots y guardado de gráficos
   - Métodos de acceso directo (.plot.line(), .plot.bar(), etc.)

### 📊 05 - Matplotlib (4 notebooks)

1. **[05.01-matplotlib-basics.ipynb](05-matplotlib/05.01-matplotlib-basics.ipynb)** - Fundamentos básicos
   - Importar Matplotlib
   - Primer gráfico básico
   - Agregar títulos y etiquetas
   - Múltiples líneas en un gráfico
   - Interface orientada a objetos (OO)
   - Guardar gráficos

2. **[05.02-matplotlib-customization.ipynb](05-matplotlib/05.02-matplotlib-customization.ipynb)** - Personalización
   - Colores (nombre, hexadecimal, RGB/RGBA)
   - Estilos de línea
   - Marcadores
   - Combinando estilos
   - Ancho de línea y transparencia
   - Personalizar ejes
   - Estilos predefinidos

3. **[05.03-matplotlib-plot-types.ipynb](05-matplotlib/05.03-matplotlib-plot-types.ipynb)** - Tipos de gráficos
   - Gráfico de barras (verticales y horizontales)
   - Gráfico de dispersión (Scatter)
   - Histogramas
   - Gráfico de área (Area Plot)
   - Gráfico de caja (Box Plot)
   - Gráfico de violín
   - Gráfico de pastel (Pie Chart)
   - Gráfico de barras agrupadas

4. **[05.04-matplotlib-advanced.ipynb](05-matplotlib/05.04-matplotlib-advanced.ipynb)** - Gráficos avanzados
   - Subplots
   - Subplots con diferentes tamaños (GridSpec)
   - Múltiples ejes (Twin Axes)
   - Gráficos 3D (superficie, línea, dispersión)
   - Anotaciones y texto
   - Líneas de referencia y regiones
   - Configuración global (rcParams)

### 📊 06 - Estadística (2 notebooks)

1. **[06.01-estadistica-basica.ipynb](06-estadistica/06.01-estadistica-basica.ipynb)** - Teoría Estadística Básica
   - Población vs Muestra
   - Tipos de datos (cualitativos, cuantitativos)
   - Medidas de tendencia central (media, mediana, moda)
   - Medidas de dispersión (varianza, desviación estándar, IQR)
   - Distribuciones de probabilidad (normal, uniforme, exponencial, etc.)
   - Teorema del Límite Central
   - Intervalos de confianza
   - Pruebas de hipótesis
   - Correlación y regresión

2. **[06.02-estadistica-aplicada-ia.ipynb](06-estadistica/06.02-estadistica-aplicada-ia.ipynb)** - Estadística Aplicada a IA
   - Estadística descriptiva con visualizaciones
   - Distribuciones de probabilidad para ML
   - Correlación y covarianza (matrices de correlación)
   - Detección de valores atípicos (outliers)
   - Intervalos de confianza
   - Pruebas de hipótesis (t-test, normalidad)
   - Normalización y estandarización para ML
   - Teorema del Límite Central aplicado

### 🤖 07 - Scikit-learn (5 notebooks)

1. **[07.01-scikit-learn-basics.ipynb](07-scikit-learn/07.01-scikit-learn-basics.ipynb)** - Fundamentos básicos
   - Importar Scikit-learn
   - Datasets integrados (iris, wine, digits, diabetes, etc.)
   - Estructura básica de trabajo con modelos
   - Flujo completo: carga → división → entrenamiento → predicción → evaluación

2. **[07.02-scikit-learn-preprocessing.ipynb](07-scikit-learn/07.02-scikit-learn-preprocessing.ipynb)** - Preprocesamiento
   - Escalado de datos (StandardScaler, MinMaxScaler, RobustScaler, Normalizer)
   - Codificación de variables categóricas (LabelEncoder, OneHotEncoder, OrdinalEncoder)
   - Manejo de valores faltantes (SimpleImputer)
   - Transformaciones polinómicas (PolynomialFeatures)
   - Pipelines de preprocesamiento

3. **[07.03-scikit-learn-supervised-learning.ipynb](07-scikit-learn/07.03-scikit-learn-supervised-learning.ipynb)** - Aprendizaje supervisado
   - **Clasificación**: LogisticRegression, DecisionTree, RandomForest, SVM, KNN, Naive Bayes, Gradient Boosting
   - **Regresión**: LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, SVR, KNN
   - Parámetros importantes y personalización de modelos

4. **[07.04-scikit-learn-unsupervised-learning.ipynb](07-scikit-learn/07.04-scikit-learn-unsupervised-learning.ipynb)** - Aprendizaje no supervisado
   - **Clustering**: K-Means, DBSCAN, Clustering Jerárquico Aglomerativo
   - **Reducción de dimensionalidad**: PCA, TruncatedSVD, NMF
   - t-SNE para visualización
   - Selección del número óptimo de componentes

5. **[07.05-scikit-learn-model-evaluation.ipynb](07-scikit-learn/07.05-scikit-learn-model-evaluation.ipynb)** - Evaluación de modelos
   - Métricas de clasificación (accuracy, precision, recall, F1, matriz de confusión)
   - Métricas de regresión (MSE, RMSE, MAE, R²)
   - Validación cruzada (K-Fold, Stratified K-Fold)
   - Búsqueda de hiperparámetros (GridSearchCV, RandomizedSearchCV)
   - Curvas ROC y AUC

### 🧩 08 - IA Clásica (Simbólica) (1 notebook)

**Nota**: Esta sección contiene algoritmos de **IA clásica/simbólica**, que son **previos al Machine Learning**. Estos algoritmos no aprenden de datos; sus reglas están prefijadas. Son conceptualmente diferentes de los algoritmos de ML, que sí aprenden de datos.

1. **[08.01-minimax.ipynb](08-ia-clasica/08.01-minimax.ipynb)** - Algoritmo Minimax
   - Fundamentos de teoría de juegos
   - Implementación básica de Minimax
   - Ejemplo práctico: Tres en Raya
   - Optimización con poda alfa-beta
   - Comparación de rendimiento
   - Aplicaciones y limitaciones
   - **Nota**: Este es un algoritmo de IA simbólica, no de Machine Learning

### 🧠 09 - Machine Learning (7 notebooks)

**Nota**: Esta sección contiene algoritmos de **Machine Learning** (que aprenden de datos), organizados por tipo de aprendizaje.

Los algoritmos están organizados por tipo de aprendizaje:

#### 📚 09.01 - Supervisados (5 notebooks)

1. **[09.01.01-arboles-decision.ipynb](09-machine-learning/09.01-supervisados/09.01.01-arboles-decision.ipynb)** - Árboles de Decisión
   - Conceptos fundamentales (Entropía, Ganancia de Información, Índice Gini)
   - Implementación para clasificación y regresión
   - Visualización de árboles de decisión
   - Importancia de características
   - Control de sobreajuste
   - Parámetros importantes

2. **[09.01.02-k-nearest-neighbors.ipynb](09-machine-learning/09.01-supervisados/09.01.02-k-nearest-neighbors.ipynb)** - K-Nearest Neighbors (KNN)
   - Algoritmo lazy learning
   - Implementación para clasificación y regresión
   - Efecto del valor de K
   - Métricas de distancia (Euclidiana, Manhattan, Minkowski)
   - Ventajas y desventajas

3. **[09.01.03-naive-bayes.ipynb](09-machine-learning/09.01-supervisados/09.01.03-naive-bayes.ipynb)** - Naive Bayes
   - Teorema de Bayes y supuesto de independencia
   - Implementación básica
   - Variantes: Gaussian, Multinomial, Bernoulli
   - Ejemplo de clasificación de texto
   - Aplicaciones en NLP

4. **[09.01.04-regresion-lineal.ipynb](09-machine-learning/09.01-supervisados/09.01.04-regresion-lineal.ipynb)** - Regresión Lineal desde Cero
   - Ecuación de regresión lineal
   - Método 1: Ecuación Normal (solución analítica)
   - Método 2: Gradiente Descendente
   - Regresión simple y múltiple
   - Visualización de convergencia

5. **[09.01.05-perceptron.ipynb](09-machine-learning/09.01-supervisados/09.01.05-perceptron.ipynb)** - Perceptrón
   - Unidad básica de redes neuronales
   - Implementación básica
   - Algoritmo de aprendizaje
   - Limitaciones (problema XOR)
   - Base para redes neuronales multicapa

#### 🔍 09.02 - No Supervisados (1 notebook)

1. **[09.02.01-k-means.ipynb](09-machine-learning/09.02-no-supervisados/09.02.01-k-means.ipynb)** - K-Means desde Cero
   - Algoritmo de clustering no supervisado
   - Implementación básica
   - Selección del número óptimo de clusters (Método del codo)
   - Métrica Silhouette Score
   - Convergencia del algoritmo

#### 🎮 09.03 - Refuerzo (1 notebook)

1. **[09.03.01-q-learning.ipynb](09-machine-learning/09.03-refuerzo/09.03.01-q-learning.ipynb)** - Q-Learning
   - Fundamentos de Reinforcement Learning
   - Ecuación de actualización Q-Learning
   - Implementación de agente Q-Learning
   - Ejemplo práctico: Laberinto
   - Visualización de tabla Q y política aprendida
   - Parámetros importantes (Learning Rate, Discount Factor, Epsilon)
   - Aplicaciones en juegos y robótica

### 🤖 10 - IA Moderna (5 notebooks)

1. **[10.01-redes-neuronales-basicas.ipynb](10-ia-moderna/10.01-redes-neuronales-basicas.ipynb)** - Redes Neuronales Básicas
   - Perceptrón Multicapa (MLP) desde cero
   - Forward propagation y backpropagation
   - Funciones de activación (sigmoid, ReLU, tanh, Leaky ReLU)
   - Ejemplo práctico: Clasificación binaria
   - Visualización de fronteras de decisión

2. **[10.02-deep-learning-tensorflow.ipynb](10-ia-moderna/10.02-deep-learning-tensorflow.ipynb)** - Deep Learning con TensorFlow/Keras
   - Construcción de modelos con Keras
   - Capas densas, dropout, batch normalization
   - Optimizadores (Adam, SGD, RMSprop)
   - Callbacks y early stopping
   - Guardar y cargar modelos
   - Transfer learning

3. **[10.03-transformers-nlp.ipynb](10-ia-moderna/10.03-transformers-nlp.ipynb)** - Transformers y NLP Moderno
   - Arquitectura Transformer
   - Attention mechanism y self-attention
   - Modelos pre-entrenados (BERT, GPT, etc.)
   - Fine-tuning de modelos
   - Hugging Face Transformers
   - Aplicaciones en NLP

4. **[10.04-cnn-convolucional.ipynb](10-ia-moderna/10.04-cnn-convolucional.ipynb)** - Redes Neuronales Convolucionales (CNN)
   - Operación de convolución desde cero
   - Pooling (Max Pooling y Average Pooling)
   - Construcción de CNN con TensorFlow/Keras
   - Clasificación de imágenes (MNIST)
   - Visualización de feature maps
   - Aplicaciones en visión por computadora

5. **[10.05-rnn-lstm.ipynb](10-ia-moderna/10.05-rnn-lstm.ipynb)** - Redes Neuronales Recurrentes (RNN) y LSTM
   - RNN básica desde cero
   - LSTM con TensorFlow/Keras
   - GRU (Gated Recurrent Unit)
   - Predicción de series temporales
   - Comparación RNN vs LSTM vs GRU
   - Aplicaciones en secuencias y NLP

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
├── 01-basicos/                      # Conocimientos básicos
│   ├── 01.01-entornos-virtuales-uv.ipynb
│   ├── 01.02-gestion-paquetes-pip.ipynb
│   ├── 01.03-comandos-terminal-bash.ipynb
│   ├── 01.04-git-basico.ipynb
│   └── 01.05-markdown.ipynb
├── 02-python/                       # Notebooks de Python
│   ├── 02.01-python-basics.ipynb
│   ├── 02.02-python-control-flow.ipynb
│   ├── 02.03-python-functions.ipynb
│   ├── 02.04-python-classes-oop.ipynb
│   └── 02.05-python-modules-io.ipynb
├── 03-numpy/                        # Notebooks de NumPy
│   ├── 03.01-numpy-basics.ipynb
│   ├── 03.02-numpy-operations.ipynb
│   ├── 03.03-numpy-indexing-slicing.ipynb
│   └── 03.04-numpy-linear-algebra.ipynb
├── 04-pandas/                       # Notebooks de Pandas
│   ├── 04.01-pandas-dataframes-series.ipynb
│   ├── 04.02-pandas-indexing-selection.ipynb
│   ├── 04.03-pandas-data-manipulation.ipynb
│   ├── 04.04-pandas-io-analysis.ipynb
│   └── 04.05-pandas-visualization.ipynb
├── 05-matplotlib/                   # Notebooks de Matplotlib
│   ├── 05.01-matplotlib-basics.ipynb
│   ├── 05.02-matplotlib-customization.ipynb
│   ├── 05.03-matplotlib-plot-types.ipynb
│   └── 05.04-matplotlib-advanced.ipynb
├── 06-estadistica/                  # Estadística
│   ├── 06.01-estadistica-basica.ipynb
│   └── 06.02-estadistica-aplicada-ia.ipynb
├── 07-scikit-learn/                 # Notebooks de Scikit-learn
│   ├── 07.01-scikit-learn-basics.ipynb
│   ├── 07.02-scikit-learn-preprocessing.ipynb
│   ├── 07.03-scikit-learn-supervised-learning.ipynb
│   ├── 07.04-scikit-learn-unsupervised-learning.ipynb
│   └── 07.05-scikit-learn-model-evaluation.ipynb
├── 08-ia-clasica/                   # IA Clásica (Simbólica) - No es ML
│   └── 08.01-minimax.ipynb
├── 09-machine-learning/             # Algoritmos Clásicos de ML
│   ├── 09.01-supervisados/          # Aprendizaje Supervisado
│   │   ├── 09.01.01-arboles-decision.ipynb
│   │   ├── 09.01.02-k-nearest-neighbors.ipynb
│   │   ├── 09.01.03-naive-bayes.ipynb
│   │   ├── 09.01.04-regresion-lineal.ipynb
│   │   └── 09.01.05-perceptron.ipynb
│   ├── 09.02-no-supervisados/       # Aprendizaje No Supervisado
│   │   └── 09.02.01-k-means.ipynb
│   └── 09.03-refuerzo/              # Aprendizaje por Refuerzo
│       └── 09.03.01-q-learning.ipynb
├── 10-ia-moderna/                   # IA Moderna
│   ├── 10.01-redes-neuronales-basicas.ipynb
│   ├── 10.02-deep-learning-tensorflow.ipynb
│   ├── 10.03-transformers-nlp.ipynb
│   ├── 10.04-cnn-convolucional.ipynb
│   └── 10.05-rnn-lstm.ipynb
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

- ✅ **43 notebooks completos** con ejemplos prácticos
- ✅ **Todos los notebooks probados** y funcionando correctamente
- ✅ **Orden lógico de aprendizaje** con numeración
- ✅ **Documentación en español**
- ✅ **Ejemplos ejecutables** sin errores
- ✅ **Kernel configurado automáticamente**
- ✅ **Contenido de IA moderna**: Transformers, CNN, RNN/LSTM, Deep Learning

## 🎯 Orden Recomendado de Aprendizaje

### Para Principiantes
1. Empieza con **01 - Básicos** (01.01-01.05) - Configuración y herramientas esenciales
2. Continúa con **02 - Python** (02.01-02.05)
3. Sigue con **03 - NumPy** (03.01-03.04)
4. Continúa con **04 - Pandas** (04.01-04.05)
5. Sigue con **05 - Matplotlib** (05.01-05.04) - Visualización de datos
6. Continúa con **06 - Estadística** (06.01-06.02) - Fundamentos estadísticos (IMPORTANTE antes de ML)
7. Sigue con **07 - Scikit-learn** (07.01-07.05) - Machine Learning
8. (Opcional) Explora **08 - IA Clásica (Simbólica)** (08.01) - Algoritmos de IA previos al ML (no aprenden de datos)
9. Explora **09 - Machine Learning** (09.01-09.05) - Algoritmos fundamentales implementados desde cero
10. Avanza a **10 - IA Moderna** (10.01-10.05) - Deep Learning y arquitecturas modernas

### Para Usuarios Avanzados
- **08 - IA Moderna**: Transformers, CNN, RNN/LSTM, Deep Learning con TensorFlow

### Para Usuarios Intermedios
- Puedes saltar directamente a la sección que necesites
- Los notebooks están numerados para facilitar la navegación

## 📚 Recursos Adicionales

- [Documentación oficial de Python](https://docs.python.org/es/3/)
- [Documentación oficial de NumPy](https://numpy.org/doc/stable/)
- [Documentación oficial de Pandas](https://pandas.pydata.org/docs/)
- [Documentación oficial de Matplotlib](https://matplotlib.org/stable/)
- [Documentación oficial de Scikit-learn](https://scikit-learn.org/stable/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [Reinforcement Learning - OpenAI Spinning Up](https://spinningup.openai.com/)
- [Teoría de Juegos - Stanford](https://web.stanford.edu/~jdlevin/Econ%20202/Game%20Theory.pdf)

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
