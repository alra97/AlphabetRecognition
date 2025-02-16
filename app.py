import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

# ✅ Load the trained model
model_path = os.path.join(os.getcwd(), "alphabet_model.h5")

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please run `train_model.py` first.")
    st.stop()
else:
    model = tf.keras.models.load_model(model_path)

# ✅ Label Mapping A-Z
labels = [chr(i) for i in range(65, 91)]  # A-Z

st.title("📝 Handwritten Alphabet Recognition")

# ✅ Upload Image
uploaded_file = st.file_uploader("Upload an image of a handwritten letter", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # ✅ Convert to black text on white background
    if np.mean(img) > 127:  # If background is white, invert colors
        img = cv2.bitwise_not(img)

    # ✅ Resize and normalize
    img = cv2.resize(img, (28, 28))
    img = img.astype("float32") / 255.0  # Normalize to [0,1]
    img = img.reshape(1, 28, 28, 1)  # Ensure correct input shape

    # ✅ Prediction
    prediction = model.predict(img)
    predicted_label = labels[np.argmax(prediction)]
    probability = np.max(prediction) * 100

    # ✅ Display results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"**Predicted Letter:** {predicted_label}")
    st.write(f"**Confidence:** {probability:.2f}%")
