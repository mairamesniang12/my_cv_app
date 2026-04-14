import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import io

st.set_page_config(page_title="Scene Classifier", page_icon="◈")

st.title("◈ Scene Classifier")
st.markdown("Upload a photo and our CNN will recognise the scene.")

# ======================
# CLASSES + EMOJIS
# ======================
CLASSES  = ["Buildings", "Forest", "Glacier", "Mountain", "Sea", "Street"]

CLASS_EMOJI = {
    "Buildings": "🏙️",
    "Forest":    "🌲",
    "Glacier":   "🧊",
    "Mountain":  "⛰️",
    "Sea":       "🌊",
    "Street":    "🛣️",
}

# ======================
# MODEL CHOICE
# ======================
model_choice = st.selectbox(
    "Choose model",
    ["pytorch", "tensorflow"]
)

# ======================
# CONFIG
# ======================
IMG_SIZE = 150
MODEL_DIR = os.path.join(os.path.dirname(__file__), "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

_loaded_models = {}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ======================
# MODEL LOADING
# ======================
def load_pytorch_model(path):
    from models.pytorch_model import IntelCNN_PyTorch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = IntelCNN_PyTorch(num_classes=len(CLASSES)).to(device)
    ckpt   = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, device

def load_tensorflow_model(path):
    import tensorflow as tf
    return tf.keras.models.load_model(path)

def get_model(model_type):
    if model_type in _loaded_models:
        return _loaded_models[model_type]

    if model_type == "pytorch":
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    else:
        candidates = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]

    if not candidates:
        st.error(f"No {model_type} model found in saved_models")
        st.stop()

    path = os.path.join(MODEL_DIR, candidates[0])

    if model_type == "pytorch":
        _loaded_models[model_type] = load_pytorch_model(path)
    else:
        _loaded_models[model_type] = load_tensorflow_model(path)

    return _loaded_models[model_type]

# ======================
# PREPROCESSING
# ======================
def preprocess(image, model_type):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0

    if model_type == "pytorch":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)
        return torch.tensor(arr).unsqueeze(0)
    else:
        return np.expand_dims(arr, axis=0)

# ======================
# PREDICTION
# ======================
def predict(image, model_type):
    model_obj = get_model(model_type)
    tensor = preprocess(image, model_type)

    if model_type == "pytorch":
        model, device = model_obj
        tensor = tensor.to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
    else:
        model = model_obj
        probs = model.predict(tensor, verbose=0)[0]

    idx = int(np.argmax(probs))
    cls = CLASSES[idx]
    confidence = float(probs[idx]) * 100

    top3 = [
        (CLASSES[i], probs[i]*100)
        for i in np.argsort(probs)[::-1][:3]
    ]

    return cls, confidence, top3

# ======================
# UI
# ======================
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Analysing..."):
            cls, conf, top3 = predict(image, model_choice)

        # RESULTAT PRINCIPAL AVEC EMOJI
        st.success(f"{CLASS_EMOJI[cls]} {cls} ({conf:.1f}%)")

        # TOP 3 AVEC EMOJIS
        st.subheader("🔝 Top 3 Predictions")
        for c, p in top3:
            st.write(f"{CLASS_EMOJI[c]} {c}: {p:.1f}%")