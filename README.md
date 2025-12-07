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

### 🐍 01 - Python (5 notebooks)

1. **[01-python-basics.ipynb](01-python/01-python-basics.ipynb)** - Fundamentos básicos
   - Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
   - Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

2. **[02-python-control-flow.ipynb](01-python/02-python-control-flow.ipynb)** - Control de flujo
   - Condicionales (if/elif/else)
   - Bucles (for, while)
   - List, Dict y Set Comprehensions
   - zip() y enumerate()

3. **[03-python-functions.ipynb](01-python/03-python-functions.ipynb)** - Funciones
   - Definición básica de funciones
   - Argumentos variables (*args, **kwargs)
   - Funciones lambda (anónimas)
   - Decoradores
   - Funciones como objetos de primera clase

4. **[04-python-classes-oop.ipynb](01-python/04-python-classes-oop.ipynb)** - Programación Orientada a Objetos (POO)
   - Clases y objetos
   - Atributos de clase vs instancia
   - Métodos de clase y estáticos
   - Herencia (simple y múltiple)
   - Métodos especiales (dunder methods)
   - Propiedades (getters y setters)
   - Encapsulación

5. **[05-python-modules-io.ipynb](01-python/05-python-modules-io.ipynb)** - Módulos e I/O
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

### 🧠 06 - Algoritmos Clásicos de Machine Learning (8 notebooks)

1. **[01-arboles-decision.ipynb](06-algoritmos-ml/01-arboles-decision.ipynb)** - Árboles de Decisión
   - Conceptos fundamentales (Entropía, Ganancia de Información, Índice Gini)
   - Implementación para clasificación y regresión
   - Visualización de árboles de decisión
   - Importancia de características
   - Control de sobreajuste
   - Parámetros importantes

2. **[02-minimax.ipynb](06-algoritmos-ml/02-minimax.ipynb)** - Algoritmo Minimax
   - Fundamentos de teoría de juegos
   - Implementación básica de Minimax
   - Ejemplo práctico: Tres en Raya
   - Optimización con poda alfa-beta
   - Comparación de rendimiento
   - Aplicaciones y limitaciones

3. **[03-q-learning.ipynb](06-algoritmos-ml/03-q-learning.ipynb)** - Q-Learning
   - Fundamentos de Reinforcement Learning
   - Ecuación de actualización Q-Learning
   - Implementación de agente Q-Learning
   - Ejemplo práctico: Laberinto
   - Visualización de tabla Q y política aprendida
   - Parámetros importantes (Learning Rate, Discount Factor, Epsilon)
   - Aplicaciones en juegos y robótica

4. **[04-k-nearest-neighbors.ipynb](06-algoritmos-ml/04-k-nearest-neighbors.ipynb)** - K-Nearest Neighbors (KNN)
   - Algoritmo lazy learning
   - Implementación para clasificación y regresión
   - Efecto del valor de K
   - Métricas de distancia (Euclidiana, Manhattan, Minkowski)
   - Ventajas y desventajas

5. **[05-naive-bayes.ipynb](06-algoritmos-ml/05-naive-bayes.ipynb)** - Naive Bayes
   - Teorema de Bayes y supuesto de independencia
   - Implementación básica
   - Variantes: Gaussian, Multinomial, Bernoulli
   - Ejemplo de clasificación de texto
   - Aplicaciones en NLP

6. **[06-regresion-lineal.ipynb](06-algoritmos-ml/06-regresion-lineal.ipynb)** - Regresión Lineal desde Cero
   - Ecuación de regresión lineal
   - Método 1: Ecuación Normal (solución analítica)
   - Método 2: Gradiente Descendente
   - Regresión simple y múltiple
   - Visualización de convergencia

7. **[07-k-means.ipynb](06-algoritmos-ml/07-k-means.ipynb)** - K-Means desde Cero
   - Algoritmo de clustering no supervisado
   - Implementación básica
   - Selección del número óptimo de clusters (Método del codo)
   - Métrica Silhouette Score
   - Convergencia del algoritmo

8. **[08-perceptron.ipynb](06-algoritmos-ml/08-perceptron.ipynb)** - Perceptrón
   - Unidad básica de redes neuronales
   - Implementación básica
   - Algoritmo de aprendizaje
   - Limitaciones (problema XOR)
   - Base para redes neuronales multicapa

### 📊 07 - Estadística (2 notebooks)

1. **[01-estadistica-basica.ipynb](07-estadistica/01-estadistica-basica.ipynb)** - Teoría Estadística Básica
   - Población vs Muestra
   - Tipos de datos (cualitativos, cuantitativos)
   - Medidas de tendencia central (media, mediana, moda)
   - Medidas de dispersión (varianza, desviación estándar, IQR)
   - Distribuciones de probabilidad (normal, uniforme, exponencial, etc.)
   - Teorema del Límite Central
   - Intervalos de confianza
   - Pruebas de hipótesis
   - Correlación y regresión

2. **[02-estadistica-aplicada-ia.ipynb](07-estadistica/02-estadistica-aplicada-ia.ipynb)** - Estadística Aplicada a IA
   - Estadística descriptiva con visualizaciones
   - Distribuciones de probabilidad para ML
   - Correlación y covarianza (matrices de correlación)
   - Detección de valores atípicos (outliers)
   - Intervalos de confianza
   - Pruebas de hipótesis (t-test, normalidad)
   - Normalización y estandarización para ML
   - Teorema del Límite Central aplicado

### 🤖 08 - IA Moderna (5 notebooks)

1. **[01-redes-neuronales-basicas.ipynb](08-ia-moderna/01-redes-neuronales-basicas.ipynb)** - Redes Neuronales Básicas
   - Perceptrón Multicapa (MLP) desde cero
   - Forward propagation y backpropagation
   - Funciones de activación (sigmoid, ReLU, tanh, Leaky ReLU)
   - Ejemplo práctico: Clasificación binaria
   - Visualización de fronteras de decisión

2. **[02-deep-learning-tensorflow.ipynb](08-ia-moderna/02-deep-learning-tensorflow.ipynb)** - Deep Learning con TensorFlow/Keras
   - Construcción de modelos con Keras
   - Capas densas, dropout, batch normalization
   - Optimizadores (Adam, SGD, RMSprop)
   - Callbacks y early stopping
   - Guardar y cargar modelos
   - Transfer learning

3. **[03-transformers-nlp.ipynb](08-ia-moderna/03-transformers-nlp.ipynb)** - Transformers y NLP Moderno
   - Arquitectura Transformer
   - Attention mechanism y self-attention
   - Modelos pre-entrenados (BERT, GPT, etc.)
   - Fine-tuning de modelos
   - Hugging Face Transformers
   - Aplicaciones en NLP

4. **[04-cnn-convolucional.ipynb](08-ia-moderna/04-cnn-convolucional.ipynb)** - Redes Neuronales Convolucionales (CNN)
   - Operación de convolución desde cero
   - Pooling (Max Pooling y Average Pooling)
   - Construcción de CNN con TensorFlow/Keras
   - Clasificación de imágenes (MNIST)
   - Visualización de feature maps
   - Aplicaciones en visión por computadora

5. **[05-rnn-lstm.ipynb](08-ia-moderna/05-rnn-lstm.ipynb)** - Redes Neuronales Recurrentes (RNN) y LSTM
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
├── 00-basicos/                      # Conocimientos básicos
│   ├── 01-entornos-virtuales-uv.ipynb
│   ├── 02-gestion-paquetes-pip.ipynb
│   ├── 03-comandos-terminal-bash.ipynb
│   ├── 04-git-basico.ipynb
│   └── 05-markdown.ipynb
├── 01-python/                       # Notebooks de Python
│   ├── 01-python-basics.ipynb
│   ├── 02-python-control-flow.ipynb
│   ├── 03-python-functions.ipynb
│   ├── 04-python-classes-oop.ipynb
│   └── 05-python-modules-io.ipynb
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
├── 06-algoritmos-ml/                # Algoritmos Clásicos de ML
│   ├── 01-arboles-decision.ipynb
│   ├── 02-minimax.ipynb
│   ├── 03-q-learning.ipynb
│   ├── 04-k-nearest-neighbors.ipynb
│   ├── 05-naive-bayes.ipynb
│   ├── 06-regresion-lineal.ipynb
│   ├── 07-k-means.ipynb
│   └── 08-perceptron.ipynb
├── 07-estadistica/                  # Estadística
│   ├── 01-estadistica-basica.ipynb
│   └── 02-estadistica-aplicada-ia.ipynb
├── 08-ia-moderna/                   # IA Moderna
│   ├── 01-redes-neuronales-basicas.ipynb
│   ├── 02-deep-learning-tensorflow.ipynb
│   ├── 03-transformers-nlp.ipynb
│   ├── 04-cnn-convolucional.ipynb
│   └── 05-rnn-lstm.ipynb
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

- ✅ **42 notebooks completos** con ejemplos prácticos
- ✅ **Todos los notebooks probados** y funcionando correctamente
- ✅ **Orden lógico de aprendizaje** con numeración
- ✅ **Documentación en español**
- ✅ **Ejemplos ejecutables** sin errores
- ✅ **Kernel configurado automáticamente**
- ✅ **Contenido de IA moderna**: Transformers, CNN, RNN/LSTM, Deep Learning

## 🎯 Orden Recomendado de Aprendizaje

### Para Principiantes
1. Empieza con **00 - Básicos** (01-05) - Configuración y herramientas esenciales
2. Continúa con **01 - Python** (01-05)
3. Sigue con **02 - NumPy** (01-04)
4. Continúa con **03 - Pandas** (01-04)
5. Sigue con **04 - Matplotlib** (01-04) - Visualización de datos
6. Continúa con **07 - Estadística** (01-02) - Fundamentos estadísticos
7. Sigue con **05 - Scikit-learn** (01-05) - Machine Learning
8. Explora **06 - Algoritmos Clásicos de ML** (01-08) - Algoritmos fundamentales implementados desde cero
9. Avanza a **08 - IA Moderna** (01-05) - Deep Learning y arquitecturas modernas

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
