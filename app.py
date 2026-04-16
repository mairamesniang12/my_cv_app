import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import base64

# ============================================
# CONFIG PAGE
# ============================================
st.set_page_config(
    page_title="Scene Classifier", 
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# STYLE
# ============================================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================
# BACKGROUND IMAGE
# ============================================
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
    except:
        pass

if os.path.exists("images/banner.jpg"):
    set_background("images/banner.jpg")

# ============================================
# HEADER
# ============================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if os.path.exists("images/banner.jpg"):
        banner = Image.open("images/banner.jpg")
        st.image(banner, width=500)
    else:
        st.title("◈ Scene Classifier")

# ============================================
# INTRO
# ============================================
st.markdown("""
### 🌍 Welcome to Scene Classifier!

This application uses a **Convolutional Neural Network (CNN)** to automatically 
recognize the type of landscape in your photos.
""")

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("### ◈ About")
    st.markdown("""
    **Scene Classifier**
    
    - 🐍 Python
    - 🔥 PyTorch / TensorFlow
    - 📊 Streamlit
    
    **Author:** Mairame Niang
    """)
    st.markdown("---")

# ============================================
# CLASSES
# ============================================
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]
CLASS_EMOJI = {
    "Buildings": "🏙️",
    "Forest": "🌲",
    "Glacier": "🧊",
    "Mountain": "⛰️",
    "Sea": "🌊",
    "Street": "🛣️",
}

# ============================================
# CONFIG
# ============================================
IMG_SIZE = 150
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ============================================
# PYTORCH MODEL
# ============================================
@st.cache_resource
def load_pytorch_model():
    from pytorch_model import IntelCNN_PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN_PyTorch(num_classes=6).to(device)

    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not model_files:
        return None, None

    model_path = os.path.join(MODEL_DIR, model_files[0])
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device

# ============================================
# TENSORFLOW MODEL
# ============================================
@st.cache_resource
def load_tensorflow_model():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
        if not model_files:
            return None

        model_path = os.path.join(MODEL_DIR, model_files[0])
        model = tf.keras.models.load_model(model_path, compile=False)
        return model

    except Exception as e:
        st.warning(f"TensorFlow model error: {e}")
        return None

# ============================================
# LOAD MODELS
# ============================================
pt_model, pt_device = load_pytorch_model()
tf_model = load_tensorflow_model()

# ============================================
# MODEL CHOICE
# ============================================
model_choice = st.selectbox(
    "Choose model",
    ["pytorch", "tensorflow"]
)

if model_choice == "pytorch" and pt_model is None:
    st.error("PyTorch model not found.")
    st.stop()

if model_choice == "tensorflow" and tf_model is None:
    st.error("TensorFlow model not found.")
    st.stop()

st.sidebar.success(f"{model_choice.upper()} model ready")

# ============================================
# PREPROCESSING
# ============================================
def preprocess_pytorch(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return torch.tensor(arr).unsqueeze(0).to(pt_device)

def preprocess_tensorflow(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(arr, axis=0)

# ============================================
# PREDICTION
# ============================================
def predict_pytorch(image):
    tensor = preprocess_pytorch(image)
    with torch.no_grad():
        logits = pt_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def predict_tensorflow(image):
    tensor = preprocess_tensorflow(image)
    probs = tf_model.predict(tensor, verbose=0)[0]
    return probs

# ============================================
# UI
# ============================================
st.markdown("---")
st.markdown("## 📤 Upload Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        if st.button("🔍 Predict"):
            with st.spinner("Analyzing..."):
                if model_choice == "pytorch":
                    probs = predict_pytorch(image)
                else:
                    probs = predict_tensorflow(image)

            idx = int(np.argmax(probs))
            cls = CLASSES[idx]
            confidence = float(probs[idx]) * 100

            top3 = [(CLASSES[i], probs[i] * 100) for i in np.argsort(probs)[::-1][:3]]

            st.markdown("### 🎯 Result")
            st.success(f"{CLASS_EMOJI[cls]} {cls} — {confidence:.1f}%")

            st.markdown("### 📊 Top 3 Predictions")
            for c, p in top3:
                st.write(f"{CLASS_EMOJI[c]} {c} — {p:.1f}%")
                st.progress(float(p) / 100)

            # Interpretation
            st.markdown("### 🧠 Confidence Level")
            if confidence > 85:
                st.success("High confidence")
            elif confidence > 60:
                st.warning("Moderate confidence")
            else:
                st.error("Low confidence")

            st.balloons()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.caption("© Scene Classifier - Computer Vision Project")
