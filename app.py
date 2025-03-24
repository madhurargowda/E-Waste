import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Load model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to classify image
def classify_image(image):
    image = image.resize((224, 224))  # Resize to model's input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Expand dims for batch size
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    prediction = np.argmax(output_data)
    labels = ["Keyboard", "Laptop_mobile", "Laptop_mouse", "Mobile", "Monitor", "Mouse"]
    return labels[prediction]

# **1️⃣ Standard Streamlit UI for File Upload**
st.title("E-Waste Image Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = classify_image(image)
    st.write(f"Prediction: **{prediction}**")

# **2️⃣ API Endpoint Simulation (For Flutter)**
st.write("### API Mode (For Flutter Integration)")

if st.button("Test API (Simulated)"):
    st.json({"message": "API is working!", "status": "success"})

# **3️⃣ Handle API Requests from Flutter**
if st.experimental_get_query_params().get("api") == ["true"]:
    uploaded_file = st.file_uploader("Upload an image via API", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        prediction = classify_image(image)
        st.json({"prediction": prediction})
