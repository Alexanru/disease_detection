# RareSight 🔬

**Early detection of rare dermatological conditions via multimodal deep learning.**

Two-stage training pipeline (MAE self-supervised pretraining → supervised fine-tuning). **Image classification + optional clinical features** for dissertation-grade research.

---

## 📚 Documentation

| What | Where |
|------|-------|
| **📑 Full map** | [INDEX.md](INDEX.md) |
| **🎓 For DL Course (April 30)** | [SUBMISSION_30APRIL.md](SUBMISSION_30APRIL.md) |
| **⚙️ What to install** | [SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md) |
| **🔄 Resume training (close laptop anytime!)** | [RESUME_TRAINING.md](RESUME_TRAINING.md) |
| **🎁 Quick setup** | [DEVBOX_QUICK.md](DEVBOX_QUICK.md) |
| **📖 Full training (Stages 1-2)** | [TRAINING_STEPS.md](TRAINING_STEPS.md) |
| **🔬 Academic info** | [ACADEMIC_DOCUMENTATION.md](ACADEMIC_DOCUMENTATION.md) |
| **🎓 Stage 3 — Multimodal (Dissertation)** | [STAGE3_MULTIMODAL.md](STAGE3_MULTIMODAL.md) |

---

## 🚀 Quick Start

**For Devbox (automatic):**
```bash
devbox shell

# Stages 1-2: Course project (by April 30)
devbox run train-s1
devbox run train-s2
# ✅ Ready for course presentation!

# Stage 3: Dissertation (April 26-30, optional)
devbox run train-s3
# ✅ Ready for dissertation!

# Or all at once:
devbox run train-all
```

**For local venv (manual):**
```bash
python -m venv venv
venv\Scripts\activate
pip install poetry==1.8.3
poetry install
python scripts/train_stage1_pretrain.py
python scripts/train_stage2_finetune.py
python scripts/train_stage3_multimodal.py  # optional
```

**For demo (no training):**
```bash
poetry install
devbox run api &
devbox run frontend
```

---

## Project Structure

**Core:**
- `src/raresight/` — PyTorch models (MAE, ViT, multimodal)
- `scripts/` — Training pipelines
- `configs/` — Hyperparameter files
- `api/` — FastAPI backend
- `frontend/` — Streamlit UI

**See [INDEX.md](INDEX.md) for complete file overview.**

---

## ⚙️ System Requirements
| **ISIC 2019** | 25,331 | Image classification (8 classes) | ✅ **Implemented** |
| HAM10000 | 10,015 | Multimodal (image + metadata) | 🔮 Future |
| PAD-UFES-20 | 2,298 | Multimodal (image + 22 features) | 🔮 Future |

**Rare classes:** Dermatofibroma (DF, 0.9%), Vascular Lesion (VASC, 1.0%), Actinic Keratosis (AK, 3.4%)

---

## 🏗️ Architecture

### Stage 1 — Masked Autoencoder (MAE)

Self-supervised pretraining on **unlabeled dermoscopy images only**. The model learns rich visual representations by reconstructing randomly masked image patches (75% masking).

- **Encoder:** ViT-Base (12 layers, 768 dim, 12 heads)
- **Decoder:** 8 layers, 512 dim
- **Loss:** MSE reconstruction on masked patches
- **No labels required** — learns from image structure alone

| Python | 3.11+  |  
| Visual Studio Code | Latest |  
| CUDA | 12.1 (for NVIDIA GPU) |  
| cuDNN | 8.x (for NVIDIA GPU) |  
| Git | 2.40+ |  

**See [SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md) for full installation guide.**

---

## 📈 Architecture Overview

**Stage 1:** Masked Autoencoder (MAE) — self-supervised pretraining on 75% masked patches  
**Stage 2:** Vision Transformer (ViT) classifier — supervised fine-tuning with focal loss  
**Optional:** Multimodal fusion (image + clinical features) for HAM10000 / PAD-UFES-20

---

## 🎯 Use Cases

✅ **For DL Lab Course (April 30):** Image classification, 2-stage training, ISIC 2019  
✅ **For Dissertation:** Multimodal fusion, comparison with literature  
✅ **For Production:** API + frontend deployment

---

## 📖 Full Guides (See [INDEX.md](INDEX.md))

- Stage-by-stage training with resume capability
- Dataset download and preprocessing
- Model evaluation and ablation study
- Docker/Rancher deployment
- API specifications
