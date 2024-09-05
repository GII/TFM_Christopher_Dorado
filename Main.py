import os
import Constantes
import tensorflow as tf
from Funciones import run_acquisition, process_images_in_folder

# Desactivar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Cargar el modelo entrenado desde el archivo guardado
model = tf.keras.models.load_model(Constantes.model_path)

# Ejecutar el programa de adquisición de fotografías
run_acquisition()

# Procesar las imágenes en la carpeta
classification_result = process_images_in_folder(Constantes.image_path, model)

# Imprimir el resultado de la clasificación
print(f'Resultado: {classification_result}')