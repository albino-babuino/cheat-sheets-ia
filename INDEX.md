# Índice de Cheat Sheets - Python, NumPy, Pandas y Matplotlib

Este repositorio contiene cheat sheets organizados por tecnología en formato Jupyter Notebook (.ipynb), ordenados numéricamente para seguir un orden lógico de aprendizaje.

## 🔧 00 - Básicos

### [01-entornos-virtuales-uv.ipynb](00-basicos/01-entornos-virtuales-uv.ipynb)
- ¿Qué es uv?
- Instalación de uv
- Crear y activar entornos virtuales
- Instalar paquetes con uv
- Gestionar dependencias
- Trabajar con proyectos

### [02-gestion-paquetes-pip.ipynb](00-basicos/02-gestion-paquetes-pip.ipynb)
- ¿Qué es pip?
- Instalar y actualizar paquetes
- Listar y buscar paquetes
- Desinstalar paquetes
- requirements.txt
- Cache y configuración

### [03-comandos-terminal-bash.ipynb](00-basicos/03-comandos-terminal-bash.ipynb)
- Navegación de directorios
- Listar archivos y directorios
- Crear y eliminar archivos/directorios
- Copiar y mover archivos
- Ver contenido de archivos
- Buscar archivos y contenido
- Permisos de archivos
- Variables de entorno
- Redirección y pipes

### [04-git-basico.ipynb](00-basicos/04-git-basico.ipynb)
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

### [05-markdown.ipynb](00-basicos/05-markdown.ipynb)
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

## 📚 01 - Python

### [01-python-basics.ipynb](01-python/01-python-basics.ipynb)
- Tipos de datos (números, strings, listas, tuplas, diccionarios, sets)
- Operadores (aritméticos, comparación, lógicos, asignación, identidad, pertenencia)

### [02-python-control-flow.ipynb](01-python/02-python-control-flow.ipynb)
- Condicionales (if/elif/else)
- Bucles (for, while)
- List Comprehensions
- Dict y Set Comprehensions
- zip() y enumerate()

### [03-python-functions-classes.ipynb](01-python/03-python-functions-classes.ipynb)
- Funciones (definición, argumentos variables, lambda)
- Clases y objetos
- Herencia
- Métodos especiales (dunder methods)
- Decoradores

### [04-python-modules-io.ipynb](01-python/04-python-modules-io.ipynb)
- Módulos y paquetes
- Manejo de archivos (I/O)
- JSON
- Manejo de excepciones

## 🔢 02 - NumPy

### [01-numpy-basics.ipynb](02-numpy/01-numpy-basics.ipynb)
- Importar NumPy
- Creación de arrays
- Propiedades de arrays
- Tipos de datos (dtype)

### [02-numpy-operations.ipynb](02-numpy/02-numpy-operations.ipynb)
- Operaciones aritméticas
- Producto matricial
- Broadcasting
- Funciones de agregación

### [03-numpy-indexing-slicing.ipynb](02-numpy/03-numpy-indexing-slicing.ipynb)
- Indexación básica
- Fancy indexing (indexación avanzada)
- Modificación de arrays
- Concatenación y división

### [04-numpy-linear-algebra.ipynb](02-numpy/04-numpy-linear-algebra.ipynb)
- Álgebra lineal (determinante, inversa, autovalores, SVD, QR)
- Estadísticas avanzadas
- Generación de números aleatorios

## 🐼 03 - Pandas

### [01-pandas-dataframes-series.ipynb](03-pandas/01-pandas-dataframes-series.ipynb)
- Crear Series
- Crear DataFrames
- Propiedades básicas

### [02-pandas-indexing-selection.ipynb](03-pandas/02-pandas-indexing-selection.ipynb)
- Selección de columnas
- Selección de filas
- Selección de filas y columnas (iloc, loc, at, iat)

### [03-pandas-data-manipulation.ipynb](03-pandas/03-pandas-data-manipulation.ipynb)
- Agregar y eliminar columnas
- Merge y Join
- GroupBy
- Pivot y Reshape

### [04-pandas-io-analysis.ipynb](03-pandas/04-pandas-io-analysis.ipynb)
- Lectura de archivos (CSV, Excel, JSON, Parquet, HTML)
- Escritura de archivos
- Análisis descriptivo
- Manejo de valores faltantes

## 📊 04 - Matplotlib

### [01-matplotlib-basics.ipynb](04-matplotlib/01-matplotlib-basics.ipynb)
- Importar Matplotlib
- Primer gráfico básico
- Agregar títulos y etiquetas
- Múltiples líneas en un gráfico
- Interface orientada a objetos (OO)
- Guardar gráficos

### [02-matplotlib-customization.ipynb](04-matplotlib/02-matplotlib-customization.ipynb)
- Colores (nombre, hexadecimal, RGB/RGBA)
- Estilos de línea
- Marcadores
- Combinando estilos
- Ancho de línea y transparencia
- Personalizar ejes
- Estilos predefinidos

### [03-matplotlib-plot-types.ipynb](04-matplotlib/03-matplotlib-plot-types.ipynb)
- Gráfico de barras (verticales y horizontales)
- Gráfico de dispersión (Scatter)
- Histogramas
- Gráfico de área (Area Plot)
- Gráfico de caja (Box Plot)
- Gráfico de violín
- Gráfico de pastel (Pie Chart)
- Gráfico de barras agrupadas

### [04-matplotlib-advanced.ipynb](04-matplotlib/04-matplotlib-advanced.ipynb)
- Subplots
- Subplots con diferentes tamaños (GridSpec)
- Múltiples ejes (Twin Axes)
- Gráficos 3D (superficie, línea, dispersión)
- Anotaciones y texto
- Líneas de referencia y regiones
- Configuración global (rcParams)

## 🚀 Uso

Para usar estos notebooks, necesitas tener instalado:
- Python 3.8+
- Jupyter Notebook o JupyterLab
- NumPy
- Pandas
- Matplotlib

### Instalación rápida con uv

```bash
# Crear entorno virtual
uv venv

# Activar entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows

# Instalar dependencias
uv pip install jupyter numpy pandas matplotlib
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
