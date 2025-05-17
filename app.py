import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# 🛠 Set page config BEFORE any other Streamlit commands
st.set_page_config(page_title="Plant Disease Detector 🌿", layout="centered")

# --- Google Drive Model Download ---
FILE_ID = "1vjUYAfoQM6pqDfzMCHFIx6Gn91hF2qeq" 
MODEL_PATH = "Plant_Disease_Detection_System.keras"

def gdrive_download_url(file_id):
    return f"https://drive.google.com/file/d/1vjUYAfoQM6pqDfzMCHFIx6Gn91hF2qeq/view?usp=drivesdk={file_id}"

@st.cache_resource
def load_plant_model():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Downloading model from Google Drive...")
        url = gdrive_download_url(FILE_ID)
        response = requests.get(url)
        if response.status_code != 200:
            st.error("❌ Failed to download model. Check your file ID or permissions.")
            st.stop()
        with open(MODEL_PATH, "wb") as f:
            f.write(response.content)
        st.success("✅ Model downloaded successfully.")
    return load_model(MODEL_PATH)

# Load model
model = load_plant_model()

# --- Class Labels ---
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- UI ---
st.title("🌿 Plant Disease Detection System")
st.write("Upload a clear image of a plant leaf, and the system will predict the disease.")

# Upload image
uploaded_file = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Leaf", use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = np.max(prediction) * 100

    # Display prediction
    st.success(f"🩺 **Prediction:** {predicted_class}")
    st.info(f"📊 **Confidence:** {confidence:.2f}%")
