import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(r'soil_model.keras')  

CLASS_NAMES = ['Black Soil', 'Cinder Soil', 'Laterite Soil', 'Peat Soil', 'Yellow Soil']

st.title("Soil Type Classifier")

file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])

if file:
    img = Image.open(file).convert('RGB').resize((224, 224))
    st.image(img, width=224)
    x = np.expand_dims(np.array(img)/255.0, axis=0)
    pred = model.predict(x)
    pred_class = CLASS_NAMES[np.argmax(pred)]
    st.write("Prediction:", pred_class)
