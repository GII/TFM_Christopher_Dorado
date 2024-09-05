import cv2
import numpy as np
from PIL import Image
import os

tile_size = 299  # Tamaño del cuadro en píxeles
dark_threshold = 30  # Umbral para determinar si una imagen es oscura
bw_threshold = 10  # Umbral para determinar si una imagen es en blanco y negro

def process_images_in_directory(src_directory: str, dst_directory: str, save):
    if not os.path.exists(src_directory):
        raise FileNotFoundError(f"Directory '{src_directory}' does not exist.")
    
    for root, dirs, files in os.walk(src_directory):
        # Crear el directorio correspondiente en el destino
        relative_path = os.path.relpath(root, src_directory)
        dest_subdir = os.path.join(dst_directory, relative_path)
        
        if not os.path.exists(dest_subdir):
            os.makedirs(dest_subdir)
        
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                processed_image = keep_leaf(image_path)

                # Obtener dimensiones de la imagen
                img_height, img_width = processed_image.shape[:2]
                
                # Inicializar contadores para nombrar los archivos
                tile_count = 0

                # Recorrer la imagen y cortar los cuadros
                for top in range(0, img_height, tile_size):
                    for left in range(0, img_width, tile_size):
                        # Definir las coordenadas del recorte
                        bottom = min(top + tile_size, img_height)
                        right = min(left + tile_size, img_width)

                        # Realizar el recorte
                        tile = processed_image[top:bottom, left:right]

                        # Evaluar si el recorte es principalmente oscuro o en blanco y negro
                        if is_dark_or_bw(tile):
                            continue  # No guardar la imagen si es principalmente oscura o en blanco y negro

                        # Convertir el recorte de BGR a RGB
                        tile_rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

                        # Convertir el recorte de nuevo a una imagen Pillow
                        tile_image = Image.fromarray(tile_rgb)

                        # Definir el nombre del archivo de salida
                        output_path = os.path.join(dest_subdir, f'{base_name}_{tile_count}.png')
                        tile_image.save(output_path)

                        # Incrementar el contador de cuadros
                        tile_count += 1

def keep_leaf(image_path):
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir el rango para el blanco intenso en HSV
    lower_white = np.array([0, 0, 220])  # Ajustar según la luminosidad y saturación del blanco
    upper_white = np.array([179, 3, 255])  # Ajustar según la luminosidad y saturación del blanco

    # Definir rangos para los colores de la hoja (verde, amarillo, naranja y marrón)
    lower_green = np.array([20, 0, 50])
    upper_green = np.array([100, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])

    lower_brown = np.array([0, 50, 50])
    upper_brown = np.array([20, 255, 255])

    # Rango para el azul
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])

    # Rango para grises
    lower_gray = np.array([0, 0, 50])
    upper_gray = np.array([179, 250, 255])

    # Crear máscaras para cada color de hoja y blanco
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Combinar las máscaras en una sola máscara de hoja
    leaf_mask = cv2.bitwise_or(green_mask, yellow_mask)
    leaf_mask = cv2.bitwise_or(leaf_mask, orange_mask)
    leaf_mask = cv2.bitwise_or(leaf_mask, brown_mask)
    leaf_mask = cv2.bitwise_or(leaf_mask, white_mask)

    # Máscaras
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # Combinar máscaras
    background_mask = cv2.bitwise_or(blue_mask, gray_mask)

    # Invertir la máscara azul para obtener el fondo (áreas que no son azul)
    background_blue_mask = cv2.bitwise_not(background_mask)

    # Crear una imagen completamente negra del mismo tamaño que la imagen original
    black_image = np.zeros_like(image)

    # Aplicar la máscara azul invertida a la imagen negra para hacer que el fondo sea negro
    background_image = cv2.bitwise_and(black_image, black_image, mask=background_blue_mask)

    # Conservar solo la hoja en la imagen original
    result = cv2.bitwise_and(image, image, mask=leaf_mask)

    # Sumar las imágenes para obtener la imagen final
    final_image = cv2.add(result, background_image)

    return final_image

def is_dark_or_bw(image, dark_threshold=dark_threshold, bw_threshold=bw_threshold):
    # Convertir la imagen a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calcular la media de los valores de los píxeles
    mean_value = np.mean(gray_image)

    # Si la media de los valores de los píxeles es menor que el umbral, la imagen es oscura
    if mean_value < dark_threshold:
        return True

    # Evaluar si la imagen es en blanco y negro
    # Calcular la desviación estándar de los valores de los canales R, G y B
    std_deviation = np.std(image, axis=(0, 1))

    # Si la desviación estándar es menor que el umbral, la imagen es en blanco y negro
    if np.max(std_deviation) < bw_threshold:
        return True

    return False

if __name__ == "__main__":
    process_images_in_directory("ImagenesAjustadas/Bueno", "ConjuntoB/Bueno", save=True)
    process_images_in_directory("ImagenesAjustadas/Mal", "ConjuntoB/Mal", save=True)
