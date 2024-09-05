import os
import gc
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import Xception, VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,BatchNormalization, Conv2D, Flatten, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize

# Desactivar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Definir la ruta de las imágenes a recortar
data_dir = './Conjunto_de_Datos'

# Leer todas las imágenes y etiquetas
image_paths = []
labels = []

for label_dir in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_dir)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(label_path, file))
                labels.append(label_dir)  # La etiqueta es el nombre de la carpeta

# Comprobar si hay imágenes cargadas
if len(image_paths) == 0:
    raise ValueError(f"No se encontraron imágenes en la ruta especificada: {data_dir}")

# Convertir a arrays de numpy
image_paths = np.array(image_paths)
labels = np.array(labels)

# Dividir en conjunto de entrenamiento y prueba (70% entrenamiento, 30% prueba)
train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels)

# Definir una función para cargar imágenes y etiquetas
def load_images(image_paths, labels, target_size=(299, 299)):  # Ajustar el tamaño de la imagen a 299x299
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images), np.array(labels)

# Cargar las imágenes y etiquetas
train_images, train_labels = load_images(train_paths, train_labels)
test_images, test_labels = load_images(test_paths, test_labels)

# Convertir etiquetas a categorías (one-hot encoding)
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(label_map))
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(label_map))

# Normalizar las imágenes
train_images = train_images / 255.0
test_images = test_images / 255.0

# Identificar la clase con menos imágenes
class_counts = {label: np.sum(train_labels[:, idx]) for label, idx in label_map.items()}
minority_class = min(class_counts, key=class_counts.get)
minority_class_idx = label_map[minority_class]

# Crear generadores de datos con aumentación solo para la clase minoritaria
train_datagen_minority = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='nearest'
)

# Filtrar imágenes de la clase minoritaria para aplicar data augmentation
minority_indices = np.where(train_labels[:, minority_class_idx] == 1)[0]
train_images_minority = train_images[minority_indices]
train_labels_minority = train_labels[minority_indices]

train_generator_minority = train_datagen_minority.flow(
    train_images_minority, train_labels_minority, batch_size=16, shuffle=True
)

# Generador para el resto de las clases sin aumentación
train_indices_non_minority = np.where(train_labels[:, minority_class_idx] == 0)[0]
train_images_non_minority = train_images[train_indices_non_minority]
train_labels_non_minority = train_labels[train_indices_non_minority]

train_generator_non_minority = ImageDataGenerator().flow(
    train_images_non_minority, train_labels_non_minority, batch_size=16, shuffle=True
)

# Combinar generadores para todas las clases
def combine_generators(gen1, gen2):
    while True:
        images1, labels1 = next(gen1)
        images2, labels2 = next(gen2)
        yield np.concatenate([images1, images2]), np.concatenate([labels1, labels2])

train_generator = combine_generators(train_generator_minority, train_generator_non_minority)

# Generador para el conjunto de prueba
test_generator = ImageDataGenerator().flow(test_images, test_labels, batch_size=16, shuffle=False)

# Callback para reducir la tasa de aprendizaje cuando la métrica se estabiliza
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)

# Callback para detener el entrenamiento temprano si no hay mejoras
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Ruta para guardar resultados
ruta_carpeta_resultados = './resultadosFinalesB/Xception'
if not os.path.exists(ruta_carpeta_resultados):
    os.makedirs(ruta_carpeta_resultados)

# Lista para guardar los resultados
resultados = []

def build_model(units, dropout, lr):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(units, activation='relu')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(len(label_map), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def create_and_train_model(base_model, model_name, input_shape=(299, 299, 3), epochs=10):
    base_model.trainable = False
    
    """# Añadir capas superiores personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)"""

    """# Usar los mejores hiperparámetros InceptionV3 en exploracion
    units = 512
    dropout = 0.6000000000000001
    lr = 0.00027407077562105383"""

    """# Usar los mejores hiperparámetros Xception en exploracion
    units = 2048
    dropout = 0.3
    lr = 3.360569327470616e-05"""

    # hiperparametros base
    units = 1024
    dropout = 0.5
    lr = 0.0001
    
    # Crear el modelo utilizando la función build_model con los mejores hiperparámetros
    model = build_model(units, dropout, lr)

    # Inicializar listas para guardar las métricas por época
    train_accs = []
    val_accs = []
    f1_scores_epoch = []
    precisions = []
    recalls = []

    for epoch in range(epochs):
        print(f'Entrenando época {epoch + 1}/{epochs} para el modelo {model_name}')
        
        # Entrenar el modelo por una época
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=test_generator,
            steps_per_epoch=len(train_images) // 32,
            validation_steps=len(test_images) // 32,
            verbose=1,  # Cambiar a verbose=1 para imprimir durante el entrenamiento
            callbacks=[reduce_lr, early_stopping]
        )
        
        # Guardar métricas de entrenamiento y validación de esta época
        train_accs.append(history.history['accuracy'][0])
        val_accs.append(history.history['val_accuracy'][0])
        
        # Calcular métricas para el conjunto de prueba
        y_pred = model.predict(test_images)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(test_labels, axis=1)
        
        # Calcular métricas y agregarlas a las listas
        f1_score_epoch = f1_score(y_true, y_pred_classes, average='weighted')
        f1_scores_epoch.append(f1_score_epoch)
        
        precision = precision_score(y_true, y_pred_classes, average='weighted', zero_division=1)
        precisions.append(precision)
        
        recall = recall_score(y_true, y_pred_classes, average='weighted')
        recalls.append(recall)

        # Guardar las métricas
        resultados.append({
            'Modelo': model_name,
            'Epoca': epoch + 1,
            'Accuracy': train_accs[-1],  # Último valor de entrenamiento
            'Val_accuracy': val_accs[-1],  # Último valor de validación
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1_score_epoch
        })

        # Guardar el reporte de clasificación en un CSV
        report = classification_report(y_true, y_pred_classes, target_names=list(label_map.keys()), output_dict=True, zero_division=1)
        report_df = pd.DataFrame(report).transpose()
        
        ruta_fold = os.path.join(ruta_carpeta_resultados, f"{model_name}_epoch_{epoch+1}")
        if not os.path.exists(ruta_fold):
            os.makedirs(ruta_fold)

        report_csv_path = os.path.join(ruta_fold, 'classification_report.csv')
        report_df.to_csv(report_csv_path)
        print(f"Reporte de clasificación guardado en {report_csv_path}")

        # Guardar las gráficas de accuracy y loss
        plt.figure()
        plt.plot(range(1, len(train_accs) + 1), train_accs, 'b', label='Training acc')
        plt.plot(range(1, len(val_accs) + 1), val_accs, 'r', label='Validation acc')
        plt.title(f'Training and Validation Accuracy - Epoch {epoch+1}')
        plt.legend()
        plt.savefig(os.path.join(ruta_fold, 'training_validation_accuracy.png'))
        plt.close()

        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(ruta_fold, 'confusion_matrix.png'))
        plt.close()

        # Calcular verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos para la matriz de confusión binaria
        if len(label_map) == 2:
            tn, fp, fn, tp = cm.ravel()
            print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")

        # Calcular y graficar la curva ROC para cada clase
        y_true_bin = label_binarize(y_true, classes=list(range(len(label_map))))
        y_pred_bin = label_binarize(y_pred_classes, classes=list(range(len(label_map))))
        plt.figure()
        for i in range(len(label_map)):
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin) #Conjunto B
            #fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i]) #Conjunto A
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f}) for label {i}')

        plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(ruta_fold, 'roc_curve.png'))
        plt.close()
        
        # Limpiar memoria después de cada época
        gc.collect()

    # Guardar el modelo
    model_path = os.path.join(ruta_carpeta_resultados, f'{model_name}_final_model')
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    # Guardar las métricas en un CSV
    res = pd.DataFrame(resultados)
    res.to_csv(os.path.join(ruta_carpeta_resultados, f'resultadosFinales.csv'), index=False)

    return model

# Modelos preentrenados
models = {
    'Xception': Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'MobileNet': MobileNet(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'EfficientNetB2': EfficientNetB2(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
    #'EfficientNetB0': EfficientNetB0(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
}

# Entrenar y guardar cada modelo, y recolectar métricas por época
for model_name, base_model in models.items():
    create_and_train_model(base_model, model_name, epochs=10)

# Limpiar memoria después del entrenamiento
gc.collect()