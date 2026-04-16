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

# STYLE
st.markdown("""
<style>
.main .block-container {
    max-width: 1400px;
    padding-top: 2rem;
}
section[data-testid="stSidebar"] {
    min-width: 300px;
}
</style>
""", unsafe_allow_html=True)

# BACKGROUND
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url(data:image/jpg;base64,{encoded_string});
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

# TITLE
st.title("◈ Scene Classifier")

# SIDEBAR
with st.sidebar:
    st.markdown("### About")
    st.write("Scene classification using CNN")

# CLASSES
CLASSES = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

# MODEL CHOICE
model_choice = st.selectbox("Choose model", ["pytorch", "tensorflow"])

# CONFIG
IMG_SIZE = 150
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_loaded_models = {}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ------------------ LOAD MODELS ------------------

def load_pytorch_model(path):
    from pytorch_model import IntelCNN_PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IntelCNN_PyTorch(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, device

def load_tensorflow_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path, compile=False)  # ✅ FIX

def get_model(model_type):
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    if model_type == "pytorch":
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    else:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]  # ✅ FIX

    if not candidates:
        st.error(f"No {model_type} model found")
        st.stop()

    path = os.path.join(MODEL_DIR, candidates[0])

    if model_type == "pytorch":
        _loaded_models[model_type] = load_pytorch_model(path)
    else:
        _loaded_models[model_type] = load_tensorflow_model(path)

    return _loaded_models[model_type]

# ------------------ PREPROCESS ------------------

def preprocess(image, model_type):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0

    if model_type == "pytorch":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr).unsqueeze(0)
    else:
        return np.expand_dims(arr, axis=0)

# ------------------ PREDICT ------------------

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

    top3 = [(CLASSES[i], probs[i]*100) for i in np.argsort(probs)[::-1][:3]]

    return cls, confidence, top3

# ------------------ UI ------------------

uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    if st.button("Predict"):
        cls, conf, top3 = predict(image, model_choice)

        st.success(f"{cls} ({conf:.1f}%)")

        st.subheader("Top 3 Predictions")
        for c, p in top3:
            st.write(f"{c}: {p:.1f}%")
