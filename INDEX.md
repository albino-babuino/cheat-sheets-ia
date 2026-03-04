# Índice de Cheat Sheets - Python, Data Science, Machine Learning e IA Moderna

Este repositorio contiene **43 cheat sheets** organizados por tecnología en formato Jupyter Notebook (.ipynb), ordenados numéricamente para seguir un orden lógico de aprendizaje. Incluye Python, NumPy, Pandas, Matplotlib, Scikit-learn, Algoritmos Clásicos de ML, Estadística e IA Moderna (Deep Learning, Transformers, CNN, RNN/LSTM).

## 🔧 01 - Básicos

### [01.01-entornos-virtuales-uv.ipynb](01-basicos/01.01-entornos-virtuales-uv.ipynb)
- ¿Qué es uv?
- Instalación de uv
- Crear y activar entornos virtuales
- Instalar paquetes con uv
- Gestionar dependencias
- Trabajar con proyectos

### [01.02-gestion-paquetes-pip.ipynb](01-basicos/01.02-gestion-paquetes-pip.ipynb)
- ¿Qué es pip?
- Instalar y actualizar paquetes
- Listar y buscar paquetes
- Desinstalar paquetes
- requirements.txt
- Cache y configuración

### [01.03-comandos-terminal-bash.ipynb](01-basicos/01.03-comandos-terminal-bash.ipynb)
- Navegación de directorios
- Listar archivos y directorios
- Crear y eliminar archivos/directorios
- Copiar y mover archivos
- Ver contenido de archivos
- Buscar archivos y contenido
- Permisos de archivos
- Variables de entorno
- Redirección y pipes

### [01.04-git-basico.ipynb](01-basicos/01.04-git-basico.ipynb)
- Configuración inicial
- Inicializar y clonar repositorios
- Estados de archivos
- Añadir y commit
- Ver historial
- Ramas (branches)
- Fusionar (merge)
- Repositorios remotos
- Push y pull
- Deshacer cambios
- .gitignore

### [01.05-markdown.ipynb](01-basicos/01.05-markdown.ipynb)
- Encabezados
- Énfasis (cursiva, negrita)
- Listas (ordenadas y no ordenadas)
- Enlaces e imágenes
- Código (inline y bloques)
- Citas (blockquotes)
- Tablas
- Líneas horizontales
- HTML inline
- Listas de tareas
- Emojis

## 📚 02 - Python

### [02.01-python-basics.ipynb](02-python/02.01-python-basics.ipynb)
- Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
- Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

### [02.02-python-control-flow.ipynb](02-python/02.02-python-control-flow.ipynb)
- Condicionales (if/elif/else)
- Bucles (for, while)
- List Comprehensions
- Dict y Set Comprehensions
- zip() y enumerate()

### [02.03-python-functions.ipynb](02-python/02.03-python-functions.ipynb)
- Funciones (definición básica, argumentos variables)
- Funciones lambda (anónimas)
- Decoradores
- Funciones como objetos de primera clase

### [02.04-python-classes-oop.ipynb](02-python/02.04-python-classes-oop.ipynb)
- Clases y objetos
- Atributos de clase vs instancia
- Métodos de clase y estáticos
- Herencia (simple y múltiple)
- Métodos especiales (dunder methods)
- Propiedades (getters y setters)
- Encapsulación

### [02.05-python-modules-io.ipynb](02-python/02.05-python-modules-io.ipynb)
- Módulos y paquetes
- Manejo de archivos (I/O)
- JSON
- Manejo de excepciones

## 🔢 03 - NumPy

### [03.01-numpy-basics.ipynb](03-numpy/03.01-numpy-basics.ipynb)
- Importar NumPy
- Creación de arrays
- Propiedades de arrays
- Tipos de datos (dtype)
- I/O (Entrada/Salida): guardar y cargar arrays (np.save, np.load, np.savetxt, np.loadtxt)

### [03.02-numpy-operations.ipynb](03-numpy/03.02-numpy-operations.ipynb)
- Operaciones aritméticas
- Producto matricial
- Broadcasting
- Funciones de agregación

### [03.03-numpy-indexing-slicing.ipynb](03-numpy/03.03-numpy-indexing-slicing.ipynb)
- Indexación básica
- Fancy indexing (indexación avanzada)
- Comparaciones elemento a elemento
- Copia vs vista (views vs copies)
- Manipulación de arrays (transpose, reshape, append, insert, delete)
- Ordenamiento de arrays

### [03.04-numpy-linear-algebra.ipynb](03-numpy/03.04-numpy-linear-algebra.ipynb)
- Álgebra lineal (determinante, inversa, autovalores, SVD, QR)
- Estadísticas avanzadas
- Generación de números aleatorios

## 🐼 04 - Pandas

### [04.01-pandas-dataframes-series.ipynb](04-pandas/04.01-pandas-dataframes-series.ipynb)
- Crear Series
- Crear DataFrames
- Propiedades básicas

### [04.02-pandas-indexing-selection.ipynb](04-pandas/04.02-pandas-indexing-selection.ipynb)
- Selección de columnas
- Selección de filas
- Selección de filas y columnas (iloc, loc, at, iat)

### [04.03-pandas-data-manipulation.ipynb](04-pandas/04.03-pandas-data-manipulation.ipynb)
- Agregar y eliminar columnas/filas
- Manipulación de índice (reindex, set_index, reset_index, sort_index)
- Ordenamiento de valores
- Merge y Join
- GroupBy (agregación, transformación, filtrado)
- Pivot y Reshape (pivot_table, melt, stack, unstack)

### [04.04-pandas-io-analysis.ipynb](04-pandas/04.04-pandas-io-analysis.ipynb)
- Lectura de archivos (CSV, Excel, JSON, Parquet, HTML)
- Escritura de archivos
- Análisis descriptivo
- Manejo de valores faltantes

### [04.05-pandas-visualization.ipynb](04-pandas/04.05-pandas-visualization.ipynb)
- Gráficos de línea (line plot)
- Gráficos de barras (bar, barh)
- Histogramas (hist)
- Gráficos de densidad (KDE)
- Box plots
- Scatter plots
- Pie charts
- Gráficos de área (area plot)
- Hexbin plots
- Personalización de gráficos (título, etiquetas, colores, tamaño, etc.)
- Subplots (múltiples gráficos)
- Selección de columnas para graficar
- Guardar gráficos
- Métodos de acceso directo (.plot.line(), .plot.bar(), etc.)

## 📊 05 - Matplotlib

### [05.01-matplotlib-basics.ipynb](05-matplotlib/05.01-matplotlib-basics.ipynb)
- Importar Matplotlib
- Primer gráfico básico
- Agregar títulos y etiquetas
- Múltiples líneas en un gráfico
- Interface orientada a objetos (OO)
- Guardar gráficos

### [05.02-matplotlib-customization.ipynb](05-matplotlib/05.02-matplotlib-customization.ipynb)
- Colores (nombre, hexadecimal, RGB/RGBA)
- Estilos de línea
- Marcadores
- Combinando estilos
- Ancho de línea y transparencia
- Personalizar ejes
- Estilos predefinidos

### [05.03-matplotlib-plot-types.ipynb](05-matplotlib/05.03-matplotlib-plot-types.ipynb)
- Gráfico de barras (verticales y horizontales)
- Gráfico de dispersión (Scatter)
- Histogramas
- Gráfico de área (Area Plot)
- Gráfico de caja (Box Plot)
- Gráfico de violín
- Gráfico de pastel (Pie Chart)
- Gráfico de barras agrupadas

### [05.04-matplotlib-advanced.ipynb](05-matplotlib/05.04-matplotlib-advanced.ipynb)
- Subplots
- Subplots con diferentes tamaños (GridSpec)
- Múltiples ejes (Twin Axes)
- Gráficos 3D (superficie, línea, dispersión)
- Anotaciones y texto
- Líneas de referencia y regiones
- Configuración global (rcParams)

## 📊 06 - Estadística

### [06.01-estadistica-basica.ipynb](06-estadistica/06.01-estadistica-basica.ipynb)
- Población vs Muestra
- Tipos de datos (cualitativos, cuantitativos)
- Medidas de tendencia central
- Medidas de dispersión
- Distribuciones de probabilidad
- Teorema del Límite Central
- Intervalos de confianza
- Pruebas de hipótesis
- Correlación y regresión

### [06.02-estadistica-aplicada-ia.ipynb](06-estadistica/06.02-estadistica-aplicada-ia.ipynb)
- Estadística descriptiva con visualizaciones
- Distribuciones de probabilidad para ML
- Correlación y covarianza
- Detección de outliers
- Intervalos de confianza
- Pruebas de hipótesis
- Normalización y estandarización
- Aplicaciones en ML/IA

## 🤖 07 - Scikit-learn

### [07.01-scikit-learn-basics.ipynb](07-scikit-learn/07.01-scikit-learn-basics.ipynb)
- Importar Scikit-learn
- Datasets integrados
- Estructura básica de trabajo con modelos
- Versión de Scikit-learn

### [07.02-scikit-learn-preprocessing.ipynb](07-scikit-learn/07.02-scikit-learn-preprocessing.ipynb)
- Escalado de datos (StandardScaler, MinMaxScaler, RobustScaler, Normalizer)
- Codificación de variables categóricas (LabelEncoder, OneHotEncoder, OrdinalEncoder)
- Manejo de valores faltantes (SimpleImputer)
- Transformaciones polinómicas (PolynomialFeatures)
- Pipeline de preprocesamiento

### [07.03-scikit-learn-supervised-learning.ipynb](07-scikit-learn/07.03-scikit-learn-supervised-learning.ipynb)
- Clasificación (LogisticRegression, DecisionTree, RandomForest, SVM, KNN, Naive Bayes, Gradient Boosting)
- Regresión (LinearRegression, Ridge, Lasso, DecisionTree, RandomForest, SVR, KNN)
- Parámetros importantes de los modelos

### [07.04-scikit-learn-unsupervised-learning.ipynb](07-scikit-learn/07.04-scikit-learn-unsupervised-learning.ipynb)
- Clustering (K-Means, DBSCAN, Clustering Jerárquico)
- Reducción de dimensionalidad (PCA, TruncatedSVD, NMF)
- t-SNE para visualización
- Selección del número de componentes

### [07.05-scikit-learn-model-evaluation.ipynb](07-scikit-learn/07.05-scikit-learn-model-evaluation.ipynb)
- Métricas de clasificación (accuracy, precision, recall, F1, matriz de confusión)
- Métricas de regresión (MSE, RMSE, MAE, R²)
- Validación cruzada (K-Fold, Stratified K-Fold)
- Búsqueda de hiperparámetros (GridSearchCV, RandomizedSearchCV)
- Curvas ROC y AUC

## 🧩 08 - IA Clásica (Simbólica) (1 notebook)

**Nota**: Esta sección contiene algoritmos de **IA clásica/simbólica**, que son **previos al Machine Learning**. Estos algoritmos no aprenden de datos; sus reglas están prefijadas. Son conceptualmente diferentes de los algoritmos de ML, que sí aprenden de datos.

### [08.01-minimax.ipynb](08-ia-clasica/08.01-minimax.ipynb)
- ¿Qué es Minimax?
- Conceptos fundamentales (Jugador MAX/MIN, árbol de juego, función de evaluación)
- Implementación básica de Minimax
- Ejemplo: Tres en Raya (Tic-Tac-Toe)
- Optimización: Poda Alfa-Beta
- Comparación: Minimax vs Minimax con Poda Alfa-Beta
- Aplicaciones y limitaciones
- **Nota**: Este es un algoritmo de IA simbólica, no de Machine Learning

## 🧠 09 - Machine Learning (7 notebooks)

**Nota**: Esta sección contiene algoritmos de **Machine Learning** (que aprenden de datos), organizados por tipo de aprendizaje.

Los algoritmos están organizados por tipo de aprendizaje:

### 📚 09.01 - Supervisados

#### [09.01.01-arboles-decision.ipynb](09-machine-learning/09.01-supervisados/09.01.01-arboles-decision.ipynb)
- ¿Qué es un Árbol de Decisión?
- Conceptos fundamentales (Entropía, Ganancia de Información, Índice Gini)
- Árbol de Decisión para Clasificación
- Árbol de Decisión para Regresión
- Visualización de árboles
- Importancia de características
- Control de sobreajuste (Overfitting)
- Parámetros importantes

#### [09.01.02-k-nearest-neighbors.ipynb](09-machine-learning/09.01-supervisados/09.01.02-k-nearest-neighbors.ipynb)
- ¿Qué es KNN?
- Conceptos fundamentales (K, Distancia, Votación, Promedio)
- Implementación básica de KNN para clasificación y regresión
- KNN para Clasificación
- KNN para Regresión
- Efecto del valor de K
- Métricas de distancia (Euclidiana, Manhattan, Minkowski)
- Ventajas, desventajas y aplicaciones

#### [09.01.03-naive-bayes.ipynb](09-machine-learning/09.01-supervisados/09.01.03-naive-bayes.ipynb)
- ¿Qué es Naive Bayes?
- Teorema de Bayes y supuesto de independencia
- Implementación básica de Naive Bayes
- Tipos de Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Ejemplo de clasificación de texto
- Comparación de variantes
- Ventajas, desventajas y aplicaciones

#### [09.01.04-regresion-lineal.ipynb](09-machine-learning/09.01-supervisados/09.01.04-regresion-lineal.ipynb)
- ¿Qué es Regresión Lineal?
- Ecuación de regresión lineal
- Método 1: Ecuación Normal (solución analítica)
- Método 2: Gradiente Descendente
- Regresión Lineal Simple
- Regresión Lineal Múltiple
- Visualización de convergencia
- Ventajas, desventajas y aplicaciones

#### [09.01.05-perceptron.ipynb](09-machine-learning/09.01-supervisados/09.01.05-perceptron.ipynb)
- ¿Qué es el Perceptrón?
- Conceptos fundamentales (Neurona artificial, Pesos, Bias, Función de activación)
- Modelo del Perceptrón y algoritmo de aprendizaje
- Implementación básica del Perceptrón
- Ejemplo de clasificación binaria
- Limitaciones (problema XOR)
- Extensión a Perceptrón Multicapa (MLP)
- Ventajas, desventajas y aplicaciones

### 🔍 09.02 - No Supervisados

#### [09.02.01-k-means.ipynb](09-machine-learning/09.02-no-supervisados/09.02.01-k-means.ipynb)
- ¿Qué es K-Means?
- Conceptos fundamentales (K, Centroide, Inicialización, Asignación, Actualización)
- Algoritmo K-Means y función de costo (Inercia)
- Implementación básica de K-Means
- Selección del número óptimo de clusters (Método del codo, Silhouette Score)
- Convergencia del algoritmo
- Ventajas, desventajas y aplicaciones

### 🎮 09.03 - Refuerzo

#### [09.03.01-q-learning.ipynb](09-machine-learning/09.03-refuerzo/09.03.01-q-learning.ipynb)
- ¿Qué es Q-Learning?
- Conceptos fundamentales (Agente, Ambiente, Estado, Acción, Recompensa, Q-Value)
- Ecuación de actualización Q-Learning
- Implementación de agente Q-Learning
- Ejemplo: Laberinto Simple
- Visualización de la tabla Q
- Parámetros importantes (Learning Rate, Discount Factor, Epsilon)
- Ventajas, desventajas y aplicaciones

## 🤖 10 - IA Moderna (5 notebooks)

### [10.01-redes-neuronales-basicas.ipynb](10-ia-moderna/10.01-redes-neuronales-basicas.ipynb)
- Perceptrón Multicapa (MLP) desde cero
- Forward propagation y backpropagation
- Funciones de activación (sigmoid, ReLU, tanh, Leaky ReLU)
- Ejemplo práctico: Clasificación binaria
- Visualización de fronteras de decisión
- Entrenamiento y optimización

### [10.02-deep-learning-tensorflow.ipynb](10-ia-moderna/10.02-deep-learning-tensorflow.ipynb)
- Construcción de modelos con Keras
- Capas densas, dropout, batch normalization
- Optimizadores (Adam, SGD, RMSprop)
- Callbacks y early stopping
- Guardar y cargar modelos
- Transfer learning
- Clasificación de imágenes (MNIST)

### [10.03-transformers-nlp.ipynb](10-ia-moderna/10.03-transformers-nlp.ipynb)
- Arquitectura Transformer
- Attention mechanism y self-attention
- Modelos pre-entrenados (BERT, GPT, etc.)
- Fine-tuning de modelos
- Hugging Face Transformers
- Análisis de sentimiento
- Generación de texto
- Traducción automática

### [10.04-cnn-convolucional.ipynb](10-ia-moderna/10.04-cnn-convolucional.ipynb)
- Operación de convolución desde cero
- Pooling (Max Pooling y Average Pooling)
- Construcción de CNN con TensorFlow/Keras
- Clasificación de imágenes (MNIST)
- Visualización de feature maps
- Arquitecturas CNN típicas
- Aplicaciones en visión por computadora

### [10.05-rnn-lstm.ipynb](10-ia-moderna/10.05-rnn-lstm.ipynb)
- RNN básica desde cero
- LSTM con TensorFlow/Keras
- GRU (Gated Recurrent Unit)
- Predicción de series temporales
- Comparación RNN vs LSTM vs GRU
- Aplicaciones en secuencias y NLP
- Normalización de datos para series temporales

## 🧠 11 - Deep Learning

### [11.01-visualizando-primera-neurona.ipynb](11-deep-learning/11.01-visualizando-primera-neurona.ipynb)
- Imagen de entrada 2x2 y normalización
- Visualización de pesos (filtro/máscara)
- Estructura de la red (4 entradas → 1 neurona)
- Función de activación ReLU
- Cálculo final: suma ponderada, sesgo y activación
- Perceptrón funcional con visualización (átomo básico del Deep Learning)

## 🚀 Uso

Para usar estos notebooks, necesitas tener instalado:
- Python 3.8+
- Jupyter Notebook o JupyterLab
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Seaborn (para algunas visualizaciones)
- TensorFlow (opcional, para notebooks de IA moderna - 10.02, 10.04, 10.05)
- Transformers y PyTorch (opcional, para notebooks de NLP - 10.03)

### Instalación rápida con uv

```bash
# Crear entorno virtual
uv venv

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Instalar dependencias
uv pip install jupyter numpy pandas matplotlib scikit-learn
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
- Los archivos están numerados para seguir un orden lógico de aprendizaje
