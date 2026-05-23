# Computer Vision — CNN (14)

Esquema **MVP** de clasificación de imágenes con **PyTorch**: dataset **CIFAR-10** (32×32, 10 clases), CNN convolucional mínima, entrenamiento y análisis en apartados separados.

## Flujo del notebook

| Paso | Contenido |
|------|-----------|
| 1 | Imports, `device`, semillas |
| 2 | Cargar CIFAR-10 (`torchvision`) y explorar clases |
| 3 | Transformaciones (`ToTensor`, normalización, augmentación en train) |
| 4 | Split **train / val** (desde train oficial) + **test** hold-out |
| 5 | `DataLoader` para train, val y test |
| 6 | Definir **CNN** (`CifarCNN`) |
| 7 | **Entrenar** (bucle épocas; solo train) |
| 8 | **Análisis**: accuracy en val y test, pérdida, reporte por clase |

Los pasos **7** y **8** están separados: el 7 solo hace `fit`; el 8 evalúa y compara.

## Notebook

| Archivo | Dataset | Modelo |
|---------|---------|--------|
| [01-cifar10-cnn-mvp.ipynb](01-cifar10-cnn-mvp.ipynb) | CIFAR-10 (50k train + 10k test) | CNN 3 bloques Conv+Pool → FC |

La primera ejecución descarga CIFAR-10 en `./data/` (caché de `torchvision`).

## Requisitos

```bash
pip install torch torchvision matplotlib scikit-learn
```

GPU opcional (`cuda` si está disponible). En CPU, reduce `EPOCHS` o `BATCH_SIZE` en el notebook.

## Material relacionado

| Tema | Carpeta |
|------|---------|
| PyTorch tabular (MLP) | [13-esquemas-sklearn-pytorch](../13-esquemas-sklearn-pytorch/) |
| Fundamentos PyTorch | [12-pytorch](../12-pytorch/) |
| Cheat sheet PyTorch | [12.00](../12-pytorch/00-pytorch-cheat-sheet.ipynb) |
