import os
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image

DATASET_PATH = "dataset"

# Generador
datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64, 64),
    color_mode="rgb",
    batch_size=4,
    class_mode="binary"
)

# Modelo
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=15)

model.save("modelo_perros_gatos_cnn.keras")

# Predicción
def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img) / 255.0
    img = img.reshape(1,64,64,3)
    pred = model.predict(img)[0][0]
    return "Gato 😺" if pred > 0.5 else "Perro 🐶"

print(predecir("dataset/gatos/cat.4001.jpg"))
