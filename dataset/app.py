import os
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.title("Clasificador de Perros y Gatos 🐶😺")

# -------------------
# Rutas
# -------------------
MODEL_PATH = "dataset/modelo_perros_gatos_cnn.keras"
DATASET_PATH = "dataset"

# -------------------
# Función de creación y entrenamiento de la CNN
# -------------------
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
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(64,64),
        batch_size=32,
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
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_gen, validation_data=val_gen, epochs=10)

    model.save(MODEL_PATH)
    st.success("Modelo entrenado y guardado ✅")
    return model

# -------------------
# Cargar modelo si existe, sino entrenar
# -------------------
if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
    model = load_model(MODEL_PATH)
    st.success("Modelo cargado ✅")
else:
    model = entrenar_modelo()

# -------------------
# Función de predicción
# -------------------
def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)[0][0]
    return "Gato 😺" if pred > 0.5 else "Perro 🐶"

# -------------------
# Interfaz Streamlit
# -------------------
st.header("Sube una imagen para predecir")
uploaded_file = st.file_uploader("Sube una imagen de un perro o un gato", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    resultado = predecir(uploaded_file)
    st.markdown(f"### Predicción: {resultado}")

# -------------------
# Predicción de todas las imágenes en carpetas
# -------------------
st.header("Predicciones de tus carpetas locales")
if st.button("Predecir todas las imágenes de perros y gatos"):
    carpetas = {"Perros": os.path.join(DATASET_PATH, "perros"),
                "Gatos": os.path.join(DATASET_PATH, "gatos")}
    
    for etiqueta, carpeta in carpetas.items():
        st.subheader(f"{etiqueta}")
        for archivo in os.listdir(carpeta):
            ruta = os.path.join(carpeta, archivo)
            pred = predecir(ruta)
            st.image(ruta, width=150, caption=f"{archivo} -> {pred}")
