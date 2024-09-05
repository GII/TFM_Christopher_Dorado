"""import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.applications import Xception, VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB2, EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import pandas as pd
from keras_tuner.tuners import RandomSearch

# Desactivar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Definir la ruta de las imágenes
data_dir = './ConjuntoBII'

# Leer todas las imágenes y etiquetas
image_paths = []
labels = []

for label_dir in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_dir)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(label_path, file))
                labels.append(label_dir)

if len(image_paths) == 0:
    raise ValueError(f"No se encontraron imágenes en la ruta especificada: {data_dir}")

image_paths = np.array(image_paths)
labels = np.array(labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels)

def load_images(image_paths, labels, target_size=(299, 299)):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images), np.array(labels)

train_images, train_labels = load_images(train_paths, train_labels)
test_images, test_labels = load_images(test_paths, test_labels)

label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(label_map))
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(label_map))

train_images = train_images / 255.0
test_images = test_images / 255.0

class_counts = {label: np.sum(train_labels[:, idx]) for label, idx in label_map.items()}
minority_class = min(class_counts, key=class_counts.get)
minority_class_idx = label_map[minority_class]

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

minority_indices = np.where(train_labels[:, minority_class_idx] == 1)[0]
train_images_minority = train_images[minority_indices]
train_labels_minority = train_labels[minority_indices]

train_generator_minority = train_datagen_minority.flow(
    train_images_minority, train_labels_minority, batch_size=16, shuffle=True
)

train_indices_non_minority = np.where(train_labels[:, minority_class_idx] == 0)[0]
train_images_non_minority = train_images[train_indices_non_minority]
train_labels_non_minority = train_labels[train_indices_non_minority]

train_generator_non_minority = ImageDataGenerator().flow(
    train_images_non_minority, train_labels_non_minority, batch_size=16, shuffle=True
)

def combine_generators(gen1, gen2):
    while True:
        images1, labels1 = next(gen1)
        images2, labels2 = next(gen2)
        yield np.concatenate([images1, images2]), np.concatenate([labels1, labels2])

train_generator = combine_generators(train_generator_minority, train_generator_non_minority)
test_generator = ImageDataGenerator().flow(test_images, test_labels, batch_size=16, shuffle=False)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ruta_carpeta_resultados = './best'
if not os.path.exists(ruta_carpeta_resultados):
    os.makedirs(ruta_carpeta_resultados)

resultados = []

def build_model(hp):
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int('units', min_value=512, max_value=2048, step=512), activation='relu')(x)
    x = Dropout(hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1))(x)
    predictions = Dense(len(label_map), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(hp.Float('lr', min_value=1e-5, max_value=1e-3, sampling='log')), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

tuner = BayesianOptimization(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='my_project')
#tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=10, executions_per_trial=1, directory='my_dir', project_name='my_project')

steps_per_epoch_train = len(train_images) // 32
steps_per_epoch_val = len(test_images) // 32

tuner.search(train_generator, epochs=10, validation_data=test_generator, steps_per_epoch=steps_per_epoch_train, validation_steps=steps_per_epoch_val)

best_model = tuner.get_best_models(num_models=1)[0]

def save_metrics(results, model_name):
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(ruta_carpeta_resultados, f'resultados_{model_name}.csv'), index=False)

def train_best_model(model, model_name, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for epoch in range(epochs):
        print(f'Entrenando época {epoch + 1}/{epochs} para el modelo {model_name}')
        
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=test_generator,
            steps_per_epoch=steps_per_epoch_train,
            validation_steps=steps_per_epoch_val,
            verbose=1,
            callbacks=[reduce_lr, early_stopping]
        )
        
        resultados.append({
            'Modelo': model_name,
            'Epoca': epoch + 1,
            'Accuracy': history.history['accuracy'][-1],
            'Val_accuracy': history.history['val_accuracy'][-1],
        })

        gc.collect()
        
    save_metrics(resultados, model_name)

train_best_model(best_model, "MobilNet_tuned", epochs=10)
"""

import os
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import pandas as pd
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.applications import Xception, VGG16, ResNet50, InceptionV3, MobileNet, EfficientNetB2, EfficientNetB0

# Desactivar GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Definir la ruta de las imágenes
data_dir = './ConjuntoBII'

# Leer todas las imágenes y etiquetas
image_paths = []
labels = []

for label_dir in os.listdir(data_dir):
    label_path = os.path.join(data_dir, label_dir)
    if os.path.isdir(label_path):
        for file in os.listdir(label_path):
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(label_path, file))
                labels.append(label_dir)

if len(image_paths) == 0:
    raise ValueError(f"No se encontraron imágenes en la ruta especificada: {data_dir}")

image_paths = np.array(image_paths)
labels = np.array(labels)

train_paths, test_paths, train_labels, test_labels = train_test_split(image_paths, labels, test_size=0.3, stratify=labels)

def load_images(image_paths, labels, target_size=(299, 299)):
    images = []
    for img_path in image_paths:
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images), np.array(labels)

train_images, train_labels = load_images(train_paths, train_labels)
test_images, test_labels = load_images(test_paths, test_labels)

label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=len(label_map))
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=len(label_map))

train_images = train_images / 255.0
test_images = test_images / 255.0

class_counts = {label: np.sum(train_labels[:, idx]) for label, idx in label_map.items()}
minority_class = min(class_counts, key=class_counts.get)
minority_class_idx = label_map[minority_class]

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

minority_indices = np.where(train_labels[:, minority_class_idx] == 1)[0]
train_images_minority = train_images[minority_indices]
train_labels_minority = train_labels[minority_indices]

train_generator_minority = train_datagen_minority.flow(
    train_images_minority, train_labels_minority, batch_size=16, shuffle=True
)

train_indices_non_minority = np.where(train_labels[:, minority_class_idx] == 0)[0]
train_images_non_minority = train_images[train_indices_non_minority]
train_labels_non_minority = train_labels[train_indices_non_minority]

train_generator_non_minority = ImageDataGenerator().flow(
    train_images_non_minority, train_labels_non_minority, batch_size=16, shuffle=True
)

def combine_generators(gen1, gen2):
    while True:
        images1, labels1 = next(gen1)
        images2, labels2 = next(gen2)
        yield np.concatenate([images1, images2]), np.concatenate([labels1, labels2])

train_generator = combine_generators(train_generator_minority, train_generator_non_minority)
test_generator = ImageDataGenerator().flow(test_images, test_labels, batch_size=16, shuffle=False)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ruta_carpeta_resultados = './best'
if not os.path.exists(ruta_carpeta_resultados):
    os.makedirs(ruta_carpeta_resultados)

resultados = []

def build_model(hp, model_name):
    models = {
        'Xception': Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'VGG16': VGG16(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'ResNet50': ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'InceptionV3': InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'MobileNet': MobileNet(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'EfficientNetB2': EfficientNetB2(weights='imagenet', include_top=False, input_shape=(299, 299, 3)),
        'EfficientNetB0': EfficientNetB0(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
    }
    
    base_model = models[model_name]
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(hp.Int('units', min_value=512, max_value=2048, step=512), activation='relu')(x)
    x = Dropout(hp.Float('dropout', min_value=0.3, max_value=0.7, step=0.1))(x)
    predictions = Dense(len(label_map), activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(hp.Float('lr', min_value=1e-5, max_value=1e-3, sampling='log')), 
                  loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_best_model(model, model_name, epochs=10):
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for epoch in range(epochs):
        print(f'Entrenando época {epoch + 1}/{epochs} para el modelo {model_name}')
        
        history = model.fit(
            train_generator,
            epochs=1,
            validation_data=test_generator,
            steps_per_epoch=len(train_images) // 32,
            validation_steps=len(test_images) // 32,
            verbose=1,
            callbacks=[reduce_lr, early_stopping]
        )
        
        resultados.append({
            'Modelo': model_name,
            'Epoca': epoch + 1,
            'Accuracy': history.history['accuracy'][-1],
            'Val_accuracy': history.history['val_accuracy'][-1],
        })

        gc.collect()
        
    save_metrics(resultados, model_name)

def save_metrics(results, model_name):
    res = pd.DataFrame(results)
    res.to_csv(os.path.join(ruta_carpeta_resultados, f'resultados_{model_name}.csv'), index=False)

# Configuración de Keras Tuner
for model_name in ['Xception', 'VGG16', 'ResNet50', 'InceptionV3', 'MobileNet', 'EfficientNetB2', 'EfficientNetB0']:
    print(f"Optimizando hiperparámetros para el modelo {model_name}...")
    
    tuner = RandomSearch(
        lambda hp: build_model(hp, model_name=model_name),
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='my_dir',
        project_name=model_name
        
    )

    tuner.search(
        train_generator,
        epochs=10,
        validation_data=test_generator,
        steps_per_epoch=len(train_images) // 32,
        validation_steps=len(test_images) // 32
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    train_best_model(best_model, model_name, epochs=10)
