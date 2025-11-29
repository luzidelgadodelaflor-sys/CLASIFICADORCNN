import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Clasificador Perros/Gatos", layout="centered")
st.title("Clasificador de Perros y Gatos 🐶😺")

# -------------------
# Cargar modelo entrenado
# -------------------
@st.cache_resource
def cargar_modelo():
    return load_model("modelo_perros_gatos_cnn.keras")

model = cargar_modelo()

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
# Interfaz
# -------------------
st.header("Sube una imagen para predecir")
uploaded_file = st.file_uploader("Sube un perro o un gato", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    resultado = predecir(uploaded_file)
    st.markdown(f"### Predicción: {resultado}")
