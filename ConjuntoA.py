"""
    Este programa sirve para recortar las imagenes que se encuentren en el directorio de entrada
    y gaurda los recortes en el directorio de salida.

    El programa no clasifica los recortes que solo tengan cinta, por lo que habrá que revisarlos manualmente
"""

import os
from PIL import Image
import numpy as np

def crop_image_into_tiles(image_path, output_folder, tile_size):
    # Obtener el nombre del archivo sin extensión
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Abrir la imagen usando Pillow
    image = Image.open(image_path)
    image_np = np.array(image)

    # Obtener dimensiones de la imagen
    img_height, img_width = image_np.shape[:2]

    # Crear la carpeta de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Inicializar contadores para nombrar los archivos
    tile_count = 0

    # Recorrer la imagen y cortar los cuadros
    for top in range(0, img_height, tile_size):
        for left in range(0, img_width, tile_size):
            # Definir las coordenadas del recorte
            bottom = min(top + tile_size, img_height)
            right = min(left + tile_size, img_width)

            # Realizar el recorte
            tile = image_np[top:bottom, left:right]

            # Convertir el recorte de nuevo a una imagen Pillow
            tile_image = Image.fromarray(tile)

            # Definir el nombre del archivo de salida
            output_path = os.path.join(output_folder, f'{base_name}_{tile_count}.png')
            tile_image.save(output_path)

            # Incrementar el contador de cuadros
            tile_count += 1

    print(f'Imagen: {image_path}, Total de cuadros guardados: {tile_count}')
    return tile_count

def process_images_in_folder(input_folder, output_folder, tile_size):
    # Inicializar contadores totales
    total_images = 0
    total_tiles = 0

    # Obtener lista de archivos en la carpeta de entrada
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            tiles_generated = crop_image_into_tiles(image_path, output_folder, tile_size)
            total_images += 1
            total_tiles += tiles_generated

    # Mostrar el total de imágenes procesadas y recortes generados
    print(f'Total de imágenes procesadas: {total_images}')
    print(f'Total de cuadros generados: {total_tiles}')

# Parámetros de entrada
input_folder = './ImagenesAjustadas/Bueno'
output_folder = './ConjuntoA/Bueno'
tile_size = 299  # Tamaño del cuadro en píxeles

# Ejecutar la función
process_images_in_folder(input_folder, output_folder, tile_size)
