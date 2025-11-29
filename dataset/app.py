import os
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

DATASET_PATH = "dataset"

datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(64,64),
    batch_size=4,
    class_mode='binary'
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5)  # menos epochs para cloud

def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Perro 🐶" if pred>0.5 else "Gato 😺"

print(predecir("dataset/gatos/cat.4001.jpg"))
