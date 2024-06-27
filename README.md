# Grupo_Pandas

YOLO TrashNet

Descripción
YOLO TrashNet es un proyecto que utiliza el modelo de detección de objetos YOLOv8 para clasificar y detectar diferentes tipos de basura en imágenes. El objetivo del proyecto es entrenar un modelo que pueda identificar diversos tipos de desechos para ayudar en la clasificación automática y eficiente de basura.

Requisitos Previos

Python 3.6+
numpy
pandas
matplotlib
opencv-python
google-colab (si usas Google Colab)
ultralytics (biblioteca YOLO)
Instalación
Sigue estos pasos para instalar y configurar el entorno de trabajo:

Clona este repositorio:


git clone https://github.com/tu_usuario/YOLO_TrashNet.git
Navega al directorio del proyecto:

sh
Copiar código
cd YOLO_TrashNet
Instala las dependencias:

sh
Copiar código
pip install -r requirements.txt
Uso
A continuación se describen los pasos para entrenar y utilizar el modelo YOLOv8 para detectar basura en imágenes.

Entrenamiento del Modelo
Monta tu Google Drive para acceder al conjunto de datos (si usas Google Colab):

python
Copiar código
from google.colab import drive
drive.mount('/content/drive')
Descomprime el conjunto de datos:

sh
Copiar código
!unzip 'drive/MyDrive/dataset.zip'
Instala la biblioteca ultralytics:

sh
Copiar código
!pip install ultralytics
Verifica que el directorio de entrenamiento existe:

python
Copiar código
import os
directory_path = '/content/train'
if os.path.exists(directory_path):
print("El directorio existe.")
else:
print("El directorio no existe.")
Carga y entrena el modelo:

python
Copiar código
from ultralytics import YOLO
model = YOLO("yolov8n.yaml") # Crear un nuevo modelo desde cero
model.train(data="config.yaml", epochs=100) # Entrenar el modelo
metrics = model.val() # Evaluar el rendimiento del modelo en el conjunto de validación
Predicción con el Modelo
Realiza predicciones en una imagen:

python
Copiar código
results = model("ruta_a_tu_imagen.jpg")
Exporta el modelo a formato ONNX:

python
Copiar código
path = model.export(format="onnx")
Muestra las predicciones en la imagen:

python
Copiar código
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
Evaluación del Modelo
Evalúa las métricas del modelo:
python
Copiar código
metrics = model.val() # Evaluar el rendimiento del modelo en el conjunto de validación
Archivar Resultados
Comprime los resultados del entrenamiento:
python
Copiar código
import shutil
folder_path = '/content/runs'
output_filename = '/content/runs.zip'
shutil.make_archive(output_filename.replace('.zip', ''), 'zip', folder_path)
