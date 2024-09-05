import os
import subprocess
import Constantes
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

# Ejecutar el script Acquisition.py
def run_acquisition():
    subprocess.run(["python", Constantes.script_path], check=False)

def process_images_in_folder(folder_path, model):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            classification_result, annotated_image = classify_grelo(image_path, model)
            if classification_result == 'Grelo Descartable':
                annotated_image.save(os.path.join(Constantes.folder_save, 'annotated_' + filename))
            return classification_result

def crop_image_into_tiles(image_path, tile_size):
    image = Image.open(image_path)
    image_np = np.array(image)
    img_height, img_width = image_np.shape[:2]

    tiles = []
    positions = []
    for top in range(0, img_height, tile_size):
        for left in range(0, img_width, tile_size):
            bottom = min(top + tile_size, img_height)
            right = min(left + tile_size, img_width)
            tile = image_np[top:bottom, left:right]
            tile_image = Image.fromarray(tile)
            tiles.append(tile_image)
            positions.append((left, top, right, bottom))

    print(f'Imagen: {image_path}, Total de cuadros generados: {len(tiles)}')
    return tiles, positions

def classify_grelo(image_path, model):
    # Obtener los recortes de la imagen
    cropped_images, positions = crop_image_into_tiles(image_path, Constantes.tile_size)
    
    if not cropped_images:
        print(f'No se generaron recortes para la imagen: {image_path}')
        return 'Desconocido', None

    # Preprocesar los recortes
    cropped_images_array = []
    for tile_image in cropped_images:
        img = img_to_array(tile_image.resize((Constantes.tile_size,Constantes.tile_size)))  # Redimensionar para que coincida con la entrada del modelo
        img = img / 255.0  # Normalizar
        cropped_images_array.append(img)
    
    cropped_images_array = np.array(cropped_images_array)
    
    # Obtener las predicciones para los recortes
    predictions = model.predict(cropped_images_array)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # Contar la cantidad de cada clase
    counts = np.bincount(predicted_classes, minlength=len(Constantes.label_map))
    print(f'Número de recuadros clasificados como "Mal": {counts[Constantes.label_map["Mal"]]}')
    print(f'Número de recuadros clasificados como "Bueno": {counts[Constantes.label_map["Bueno"]]}')
    
    # Crear una copia de la imagen original para anotaciones
    original_image = Image.open(image_path)
    annotated_image = original_image.copy()
    draw = ImageDraw.Draw(annotated_image)
    
    # Anotar los recortes clasificados como "Mal"
    for idx, pos in enumerate(positions):
        if predicted_classes[idx] == Constantes.label_map['Mal']:
            draw.rectangle(pos, outline="red", width=5)
    
    # Verificar si hay alguna predicción como 'Mal'
    if counts[Constantes.label_map['Mal']] > 0:
        return 'Grelo Descartable', annotated_image
    # Verificar si hay alguna predicción como 'Bueno'
    elif counts[Constantes.label_map['Bueno']] > 0:
        return 'Grelo Bueno', None
    else:
        return 'Desconocido', None