import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

st.set_page_config(page_title="Clasificador Perros/Gatos", layout="wide")
st.title("Clasificador de Perros y Gatos 🐶😺")

DATASET_PATH = "dataset"  # Debe contener carpetas 'perros' y 'gatos'

# -------------------
# Entrenamiento de la CNN
# -------------------
@st.cache_resource(show_spinner=False)
def entrenar_modelo():
    st.info("Entrenando CNN desde cero... Esto puede tardar unos minutos ⏳")

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 horizontal_flip=True, rotation_range=20)

    train_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64,64),
        batch_size=4,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64,64),
        batch_size=4,
        class_mode='binary',
        subset='validation'
    )

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
    model.fit(train_gen, validation_data=val_gen, epochs=3, verbose=1)

    st.success("Modelo entrenado ✅")
    return model

model = entrenar_modelo()

# -------------------
# Función de predicción
# -------------------
def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img, verbose=0)[0][0]
    return "Perro 🐶" if pred>0.5 else "Gato 😺"

# -------------------
# Predicción imagen subida
# -------------------
st.header("Sube una imagen para predecir")
uploaded_file = st.file_uploader("Sube un perro o un gato", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    resultado = predecir(uploaded_file)
    st.markdown(f"### Predicción: {resultado}")

# -------------------
# Predicción de carpetas locales
# -------------------
st.header("Predicciones de tus carpetas locales")
if st.button("Predecir todas las imágenes de perros y gatos"):
    carpetas = {"Perros": os.path.join(DATASET_PATH, "perros"),
                "Gatos": os.path.join(DATASET_PATH, "gatos")}

    for etiqueta, carpeta in carpetas.items():
        st.subheader(etiqueta)
        if os.path.exists(carpeta):
            columnas = st.columns(3)  # Mostrar 3 imágenes por fila
            idx = 0
            for archivo in os.listdir(carpeta):
                if archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                    ruta = os.path.join(carpeta, archivo)
                    pred = predecir(ruta)
                    columnas[idx % 3].image(ruta, width=150, caption=f"{archivo} -> {pred}")
                    idx += 1
        else:
            st.warning(f"No se encontró la carpeta {carpeta}")
