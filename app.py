import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import base64

st.set_page_config(
    page_title="Scene Classifier", 
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BACKGROUND IMAGE
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/{"jpg" if image_path.endswith('.jpg') else "png"};base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        return True
    except:
        return False

background_path = "images/banner.jpg"  
if os.path.exists(background_path):
    set_background(background_path)

# HEADER
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    try:
        banner = Image.open("images/banner.jpg")
        st.image(banner, width=500)
    except:
        st.title("◈ Scene Classifier")

# MAIN CONTENT
st.markdown("""
### 🌍 Welcome to Scene Classifier!

This application uses a **Convolutional Neural Network (CNN)** to automatically 
recognize the type of landscape in your photos.

---
""")

# Sidebar
with st.sidebar:
    st.markdown("### ◈ About")
    st.markdown("""
    **Scene Classifier** - Computer Vision Project
    
    - 🐍 Python 3.10
    - 🔥 PyTorch
    - 📊 Streamlit
    
    **Author:** Mairame Niang
    
    **Version:** 1.0.0
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **Framework:** PyTorch
    - **Accuracy:** 90.47%
    - **Parameters:** 2.68M
    """)

# CLASSES
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
CLASS_EMOJI = {
    "Buildings": "🏙️",
    "Forest": "🌲",
    "Glacier": "🧊",
    "Mountain": "⛰️",
    "Sea": "🌊",
    "Street": "🛣️",
}

# CONFIG
IMG_SIZE = 150
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MODEL LOADING - PYTORCH ONLY
@st.cache_resource
def load_model():
    from pytorch_model import IntelCNN_PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN_PyTorch(num_classes=6).to(device)
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not model_files:
        st.error(f"No model found in {MODEL_DIR}")
        st.stop()
    model_path = os.path.join(MODEL_DIR, model_files[0])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

model, device = load_model()
st.sidebar.success("✅ Model loaded successfully!")

# PREPROCESSING
def preprocess(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)

# PREDICTION
def predict(image):
    tensor = preprocess(image)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    cls = CLASSES[idx]
    confidence = float(probs[idx]) * 100
    top3 = [(CLASSES[i], probs[i] * 100) for i in np.argsort(probs)[::-1][:3]]
    return cls, confidence, top3

# UI
st.markdown("---")
st.markdown("## 📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"],
    help="Accepted formats: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(image, caption="📷 Uploaded image", use_container_width=True)
    with col2:
        if st.button("🔍 Predict", type="primary", use_container_width=True):
            with st.spinner("🧠 Analyzing..."):
                cls, conf, top3 = predict(image)
            st.markdown("---")
            st.markdown("### 🎯 Main Result")
            st.success(f"{CLASS_EMOJI[cls]} **{cls}** ({conf:.1f}% confidence)")
            st.markdown("### 📊 Top 3 Predictions")
            for c, p in top3:
                st.progress(p/100, text=f"{CLASS_EMOJI[c]} {c}: {p:.1f}%")
            st.balloons()

st.markdown("---")
st.caption("© 2024 Scene Classifier - Computer Vision Project | Accuracy: 90.47%")
