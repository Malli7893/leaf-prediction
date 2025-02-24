import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

file_id="1em_4pDcMztrkxm1gH0WWmf9I8brKAzkj"
url='https://drive.google.com/file/d/1em_4pDcMztrkxm1gH0WWmf9I8brKAzkj/view?usp=sharing'
model_path='trained_plant_disease_model.keras'

if not os.path.exists(model_path):
    st.warning("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)
model = tf.keras.models.load_model(model_path)

# Define class labels for potato leaf diseases
class_labels = ['Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy']

# Custom CSS for styling with animated 3D background
st.markdown(
    """
    <style>
        @keyframes gradientBG {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        body, .stApp {
            background: linear-gradient(45deg, #ff9966, #ff5e62, #6a11cb, #2575fc);
            background-size: 400% 400%;
            animation: gradientBG 10s ease infinite;
            color: white;
        }
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        @keyframes floating {
            0% { transform: translateY(0px); }
            50% { transform: translateY(-10px); }
            100% { transform: translateY(0px); }
        }
        .floating {
            animation: floating 3s infinite ease-in-out;
        }
        .marquee {
            width: 100%;
            overflow: hidden;
            white-space: nowrap;
            box-sizing: border-box;
            animation: marquee 10s linear infinite;
        }
        @keyframes marquee {
            from { transform: translateX(100%); }
            to { transform: translateX(-100%); }
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.markdown("<h1 class='floating'>🥔 Potato Leaf Disease Classification</h1>", unsafe_allow_html=True)
st.write("Upload an image of a potato leaf to classify its disease.")

# File uploader with smaller size
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=False)
    
    # Ensure image is in RGB mode
    image = image.convert("RGB")
    
    # Preprocess the image
    image = image.resize((128, 128))  # Resize to match model input size
    image_array = np.array(image)  # Keep raw pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    
    # Make prediction
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Get highest probability class
    confidence = np.max(predictions)  # Get confidence score
    
    # Display prediction results with animated marquee text
    result_text = f"Predicted Class: {class_labels[predicted_class]} | Confidence: {confidence:.2f}"
    st.markdown(f"<h2>{result_text}</h2></div>", unsafe_allow_html=True)
    
    # Display additional message based on prediction
    if class_labels[predicted_class] == 'Potato_Early_blight':
        st.warning("⚠ This leaf has Early Blight. Consider using fungicides and improving field management.")
    elif class_labels[predicted_class] == 'Potato_Late_blight':
        st.error("🚨 This leaf has Late Blight. Immediate action is needed to prevent crop loss!")
    else:
        st.success("✅ This potato leaf is healthy!")
