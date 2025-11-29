import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Clasificador de Perros y Gatos 🐶😺")

# -------------------
# Cargar modelo CNN ya entrenado
# -------------------
MODEL_PATH = "modelo_perros_gatos_cnn.keras"
model = load_model(MODEL_PATH)
st.success("Modelo cargado ✅")

# -------------------
# Función de predicción
# -------------------
def predecir(ruta):
    img = Image.open(ruta).resize((64,64)).convert("RGB")
    img = np.array(img) / 255.0
    img = img.reshape(1,64,64,3)
    pred = model.predict(img)[0][0]
    return "Gato 😺" if pred > 0.5 else "Perro 🐶"

# -------------------
# Interfaz de Streamlit
# -------------------
uploaded_file = st.file_uploader("Sube una imagen de un perro o un gato", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    # Guardar temporalmente
    image_path = "temp.jpg"
    image.save(image_path)
    
    # Predicción
    resultado = predecir(image_path)
    st.markdown(f"### Predicción: {resultado}")

