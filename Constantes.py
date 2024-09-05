"""
    Archivo donde se definen las rutas y las variables que son constantes en el uso del programa
"""

# Direccion del directorio donde se guardan las imagenes captadas por la cámara
image_path = 'C:/Users/96_ch/Desktop/Master/TFM/Github/Imagenes' 

# Direccion del directorio donde se guarda el modelo entrenado
model_path = 'C:/Users/96_ch/Desktop/Master/TFM/Github/Modelos/InceptionV3_initial_model'

# Direccion del programa para disparar una fotografía en la cámara de Flir
script_path = 'C:/Users/96_ch/Desktop/Master/TFM/Github/Acquisition.py'

# Tamaño del cuadro en píxeles
tile_size = 299

# Definir el valor de las etiquetas
label_map = {'Bueno': 0, 'Cinta': 1, 'Mal': 2}

# Ruta de destino para guardar la imagen segmentada
folder_save = 'C:/Users/96_ch/Desktop/Master/TFM/Github/Imagen_segmentada'
