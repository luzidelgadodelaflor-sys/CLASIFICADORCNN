import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.title("Clasificador de Perros y Gatos 🐶😺")

DATASET_PATH = "dataset"  # Debe tener subcarpetas 'perros' y 'gatos'

def entrenar_modelo():
    st.info("Entrenando CNN desde cero... Esto puede tardar varios minutos ⏳")

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        rotation_range=20
    )

    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64,64),
        batch_size=16,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64,64),
        batch_size=16,
        class_mode='binary',
        subset='validation'
    )

    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D(2,2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=3)  # menos epochs para cloud

    st.success("Modelo entrenado ✅")
    return model

model = entrenar_modelo()

def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Gato 😺" if pred>0.5 else "Perro 🐶"

st.header("Sube una imagen para predecir")
uploaded_file = st.file_uploader("Sube una imagen de un perro o gato", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    st.markdown(f"### Predicción: {predecir(uploaded_file)}")
