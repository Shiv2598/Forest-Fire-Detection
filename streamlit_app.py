import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

st.title("🌲 Forest Fire Detection App")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("forest_fire_detection_model.h5")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload a forest image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    img_array = np.array(image.resize((64, 64)))  # Assuming model expects 64x64 input
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("🔥 Fire Detected!")
    else:
        st.success("✅ No Fire Detected!")
