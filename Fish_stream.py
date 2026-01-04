import streamlit as st
import tensorflow as tf
import numpy as np
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Load Model and Class Indices
# -----------------------------

MODEL_PATH = "Fish_model.h5"
CLASS_INDEX_PATH = "class_indices.json"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class index mapping
with open(CLASS_INDEX_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index → class name
index_to_class = {v: k for k, v in class_indices.items()}

IMG_SIZE = 224

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🐟 Multiclass Fish Image Classification")
st.write("Upload a fish image and the model will predict its species.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Display uploaded image
    image = load_img(uploaded_file, target_size=(IMG_SIZE, IMG_SIZE))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = img_to_array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    predicted_class = index_to_class[predicted_index]
    confidence = float(np.max(predictions[0])) * 100

    # Output
    st.subheader("Prediction")
    st.write(f"**Predicted Species:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.subheader("Class Probabilities")
    for idx, prob in enumerate(predictions[0]):
        st.write(f"{index_to_class[idx]}: {prob*100:.2f}%")
