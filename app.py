import streamlit as st
import torch
import base64
import numpy as np
from PIL import Image
import os
import io

st.set_page_config(
    page_title="Scene Classifier", 
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Élargir le contenu principal */
    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Élargir la sidebar si nécessaire */
    section[data-testid="stSidebar"] {
        min-width: 300px;
        width: 300px;
    }
    
    /* Meilleure utilisation de l'espace */
    .stMarkdown {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# BACKGROUND IMAGE WITH CSS
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: #000000;
                background-image: url(data:image/{"jpg" if image_path.endswith('.jpg') else "png"};base64,{encoded_string});
                background-size: contain;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Ajouter une légère transparence pour la lisibilité */
            .stMarkdown, .stTitle, .stSubheader, .stText, .stSelectbox label {{
                background-color: rgba(0, 0, 0, 0.6);
                padding: 10px;
                border-radius: 10px;
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        return True
    except:
        return False


col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    try:
        banner = Image.open("images/banner.jpg")
        st.image(banner, width=600)  
    except:
        pass 

# APP DESCRIPTION
st.title("◈ Scene Classifier")
st.markdown("---")

# Main description
st.markdown("""
### 🌍 Welcome to Scene Classifier!

This application uses a **Convolutional Neural Network (CNN)** to automatically 
recognize the type of landscape in your photos.

#### 🎯 What our model can recognize:
- 🏙️ **Buildings** - urban areas, cityscapes, architecture
- 🌲 **Forests** - woodlands, trees, natural landscapes
- 🧊 **Glaciers** - snowy mountains, ice, snow-covered peaks
- ⛰️ **Mountains** - rocky landscapes, mountain ranges
- 🌊 **Sea** - oceans, beaches, lakes, water bodies
- 🛣️ **Streets** - roads, avenues, paved areas

#### 🚀 How to use the application:
1. **Choose the model** (PyTorch or TensorFlow)
2. **Upload an image** (JPG, PNG, JPEG formats)
3. **Click "Predict"**
4. **Get results** with top 3 predictions

#### 📊 Model Performance:
- Custom CNN architecture with over **2.6 million parameters**
- Trained on Intel Image Classification dataset
- Validation accuracy: **~85-90%**

---
""", unsafe_allow_html=False)

# Sidebar with information
with st.sidebar:
    st.markdown("### ◈ About")
    st.markdown("""
    **Scene Classifier** is a computer vision project developed with:
    
    - 🐍 Python 3.9
    - 🔥 PyTorch / TensorFlow
    - 📊 Streamlit
    
    ---
    
    **Author:** Mairame Niang
    
    **Version:** 1.0.0
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Use clear, well-lit images
    - Prefer landscape-oriented photos
    - Avoid overly dark or blurry images
    """)
    
    st.markdown("---")
    st.markdown("### 📁 Available Models")
    st.markdown("""
    - **PyTorch** : `.pth` (2.6M parameters)
    - **TensorFlow** : `.h5` (2.6M parameters)
    """)

# CLASSES + EMOJIS
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

CLASS_EMOJI = {
    "Buildings": "🏙️",
    "Forest":    "🌲",
    "Glacier":   "🧊",
    "Mountain":  "⛰️",
    "Sea":       "🌊",
    "Street":    "🛣️",
}


# MODEL CHOICE
model_choice = st.selectbox(
    "Choose model",
    ["pytorch", "tensorflow"]
)


# CONFIG
IMG_SIZE = 150
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

_loaded_models = {}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# MODEL LOADING
def load_pytorch_model(path):
    from pytorch_model import IntelCNN_PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN_PyTorch(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device

# ✅ CORRECTION : chargement du modèle .h5 au lieu de .keras
def load_tensorflow_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)

def get_model(model_type):
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    if model_type == "pytorch":
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    else:
        # ✅ CORRECTION : chercher .h5 au lieu de .keras
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]

    if not candidates:
        st.error(f"No {model_type} model found in {MODEL_DIR}")
        st.stop()

    path = os.path.join(MODEL_DIR, candidates[0])
    st.sidebar.success(f"✅ {model_type.upper()} model: {os.path.basename(path)}")

    if model_type == "pytorch":
        _loaded_models[model_type] = load_pytorch_model(path)
    else:
        _loaded_models[model_type] = load_tensorflow_model(path)

    return _loaded_models[model_type]


# PREPROCESSING
def preprocess(image, model_type):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0

    if model_type == "pytorch":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    else:
        return np.expand_dims(arr, axis=0)


# PREDICTION
def predict(image, model_type):
    model_obj = get_model(model_type)
    tensor = preprocess(image, model_type)

    if model_type == "pytorch":
        model, device = model_obj
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        model = model_obj
        probs = model.predict(tensor, verbose=0)[0]

    idx = int(np.argmax(probs))
    cls = CLASSES[idx]
    confidence = float(probs[idx]) * 100

    top3 = [
        (CLASSES[i], probs[i] * 100)
        for i in np.argsort(probs)[::-1][:3]
    ]

    return cls, confidence, top3


# UI
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analysing..."):
            cls, conf, top3 = predict(image, model_choice)

        # Main result with emoji
        st.success(f"{CLASS_EMOJI[cls]} {cls} ({conf:.1f}%)")

        # Top 3 with emojis
        st.subheader("🔝 Top 3 Predictions")
        for c, p in top3:
            st.write(f"{CLASS_EMOJI[c]} {c}: {p:.1f}%")
