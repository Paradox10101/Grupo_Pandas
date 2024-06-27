# Grupo - Pandas

## Índice
- [Integrantes](#integrantes)
- [YOLO TrashNet](#yolo-trashnet)
  - [Descripción](#descripción)
  - [Requisitos Previos](#requisitos-previos)
  - [Instalación](#instalación)
  - [Uso](#uso)
    - [Entrenamiento del Modelo](#entrenamiento-del-modelo)
    - [Predicción con el Modelo](#predicción-con-el-modelo)
    - [Evaluación del Modelo](#evaluación-del-modelo)
  - [Archivar Resultados](#archivar-resultados)


## Integrantes
- Edgardo Solis
- Charlie Levano
- Maria Rojas
- Alexandra Zavala

## YOLO TrashNet

### Descripción
YOLO TrashNet es un proyecto que utiliza el modelo de detección de objetos YOLOv8 para clasificar y detectar diferentes tipos de basura en imágenes. El objetivo del proyecto es entrenar un modelo que pueda identificar diversos tipos de desechos para ayudar en la clasificación automática y eficiente de basura.

### Requisitos Previos
- Python 3.6+
- numpy
- pandas
- matplotlib
- opencv-python
- google-colab (si usas Google Colab)
- ultralytics (biblioteca YOLO)

### Instalación
Sigue estos pasos para instalar y configurar el entorno de trabajo:

1. Clona este repositorio:
    ```sh
    git clone https://github.com/tu_usuario/YOLO_TrashNet.git
    ```

2. Navega al directorio del proyecto:
    ```sh
    cd YOLO_TrashNet
    ```

3. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

### Uso
A continuación se describen los pasos para entrenar y utilizar el modelo YOLOv8 para detectar basura en imágenes.

#### Entrenamiento del Modelo
1. Monta tu Google Drive para acceder al conjunto de datos (si usas Google Colab):
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. Descomprime el conjunto de datos:
    ```sh
    !unzip 'drive/MyDrive/dataset.zip'
    ```

3. Instala la biblioteca ultralytics:
    ```sh
    !pip install ultralytics
    ```

4. Verifica que el directorio de entrenamiento existe:
    ```python
    import os
    directory_path = '/content/train'
    if os.path.exists(directory_path):
        print("El directorio existe.")
    else:
        print("El directorio no existe.")
    ```

5. Carga y entrena el modelo:
    ```python
    from ultralytics import YOLO
    model = YOLO("yolov8n.yaml") # Crear un nuevo modelo desde cero
    model.train(data="config.yaml", epochs=100) # Entrenar el modelo
    metrics = model.val() # Evaluar el rendimiento del modelo en el conjunto de validación
    ```

#### Predicción con el Modelo
1. Realiza predicciones en una imagen:
    ```python
    results = model("ruta_a_tu_imagen.jpg")
    ```

2. Exporta el modelo a formato ONNX:
    ```python
    path = model.export(format="onnx")
    ```

3. Muestra las predicciones en la imagen:
    ```python
    import matplotlib.pyplot as plt
    import cv2
    if results and isinstance(results, list):
        for result in results:
            img_with_boxes = result.plot()
            img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            plt.imshow(img_with_boxes_rgb)
            plt.axis('off')
            plt.show()
    else:
        print("No se encontraron resultados.")
    ```

#### Evaluación del Modelo
1. Evalúa las métricas del modelo:
    ```python
    metrics = model.val() # Evaluar el rendimiento del modelo en el conjunto de validación
    ```

### Archivar Resultados
1. Comprime los resultados del entrenamiento:
    ```python
    import shutil
    folder_path = '/content/runs'
    output_filename = '/content/runs.zip'
    shutil.make_archive(output_filename.replace('.zip', ''), 'zip', folder_path)
    ```
