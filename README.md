# MairaNet — Intel Image Classification
**Author: Mairame**

# ◈ Scene Classifier — MairaNet

**Author:** Mairame Niang

A complete deep learning pipeline for the [Intel Image Classification dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification), featuring two original CNN architectures (PyTorch + TensorFlow) served via a **Streamlit** web application.

---

## 📊 Dataset

Download from Kaggle and unzip so the folder layout is:


---

## Dataset

Download from Kaggle and unzip so the folder layout is:

```
data/
  seg_train/seg_train/
    buildings/  forest/  glacier/  mountain/  sea/  street/
  seg_test/seg_test/
    buildings/  forest/  glacier/  mountain/  sea/  street/
```

**6 classes · 14,034 train images · 3,000 test images · 150×150 px**

---

## Project structure

```
PROJECT1/
├── app.py # Streamlit web application
├── pytorch_model.py # MairaNet-PT architecture (PyTorch)
├── tensorflow_model.py # MairaNet-TF architecture (TensorFlow/Keras)
├── train.py # Unified training script
├── requirements.txt # Python dependencies
├── models/ # Saved weights after training
│ ├── mairame_model.pth # PyTorch weights
│ └── mairame_model.keras # TensorFlow weights
├── images/ # App assets (background, banner)
└── data/ # Dataset
```

---

## Dependencies

Python **3.9+** required.

```
numpy>=1.23.5
torch>=2.0.1
torchvision>=0.15.2
tensorflow>=2.15.0
streamlit>=1.28.0
pillow>=9.0.0
```

Install:
```bash
pip install -r requirements.txt
```

---

## Training

The `--model` argument selects which framework to use. GPU is detected and used automatically.

```bash
# Train PyTorch model → saves models/mairame_model.pth
python train.py --model pytorch --data_dir ./data --epochs 15 --batch_size 64 --lr 0.001

# Train TensorFlow model → saves models/mairame_model.keras
python train.py --model tensorflow --data_dir ./data --epochs 15 --batch_size 64 --lr 0.001
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | **required** | `pytorch` or `tensorflow` |
| `--data_dir` | `./data` | Root of the Intel dataset |
| `--epochs` | `15` | Training epochs |
| `--batch_size` | `64` | Batch size |
| `--lr` | `1e-3` | Initial learning rate |
| `--img_size` | `150` | Input resolution (px) |
| `--output_dir` | `./models` | Where to write weights |

---

## Architecture summary

### MairaNet-PT (PyTorch)
- 4 stages of double Conv2d + BatchNorm + ReLU
- Global Average Pooling → Dropout(0.4) → Dense(512) → Dropout(0.3) → Dense(6)
- ~2.7M parameters

### MairaNet-TF (TensorFlow/Keras)
- Similar architecture with depthwise-separable convolutions
- ~0.7M parameters (lighter)

### Training strategy (both models)
| | PyTorch | TensorFlow |
|---|---|---|
| Optimiser | AdamW (wd=1e-4) | AdamW (wd=1e-4) |
| LR schedule | OneCycleLR | CosineDecayRestarts |
| Loss | CE + label smoothing 0.1 | SparseCCE (from logits) |
| Early stopping | — | patience=10 on val_acc |
| Checkpointing | Best val_acc | Best val_acc |

---

## Running locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train (or place pre-trained weights in ./models/)
python train.py --model pytorch --data_dir ./data

# 3. Start the Streamlit app
streamlit run app.py

# 4. Open http://localhost:8501
```

---

## Deployment

### Streamlit Cloud (Recommended - Free)
```bash
1.Push your code to GitHub
2.Go to share.streamlit.io
3.Sign in with GitHub
4.Select your repository and branch
5.Click "Deploy"
```
---

## Web interface features

- **Model selection** — switch between MairaNet-PT (PyTorch) and MairaNet-TF (TensorFlow)
- **Image upload** — Drag & drop or browse (JPG, PNG, JPEG)
- **Real-time prediction** — Instant results with confidence scores
- **Top 3 predictions** — Visual progress bars for each class
- **Responsive design** — Custom background and modern UI
