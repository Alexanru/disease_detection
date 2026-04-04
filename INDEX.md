# 📑 RareSight — Complete Index

**Find exactly what you need:**

---

## 🚀 Getting Started (Pick One Path)

1. **[🎁 DEVBOX_QUICK.md](DEVBOX_QUICK.md)** — Auto-setup + separate commands
   - `devbox run setup` → `devbox run train` → `devbox run api`
   - Best for: Beginners, quick setup

2. **[📖 TRAINING_STEPS.md](TRAINING_STEPS.md)** — Step-by-step manual training
   - Full local venv setup + data download + training
   - Best for: Full control, understanding each step

3. **[🐳 DOCKER_RANCHER.md](DOCKER_RANCHER.md)** — Container orchestration
   - Docker + Rancher UI + Kubernetes
   - Best for: Production, multi-machine deployment

4. **[⚡ QUICK_START.md](QUICK_START.md)** — Try without training
   - 5-minute demo with mock model
   - Best for: Testing UI/API without GPU

---

## 📚 Documentation

### For Academia (Dissertations)

👉 **[📚 ACADEMIC_DOCUMENTATION.md](ACADEMIC_DOCUMENTATION.md)**

- 1. Problem Statement (rare disease detection)
- 2. Literature Review (MAE, ViT, Focal Loss)
- 3. Methodology (architecture, training details)
- 4. Experimental Setup & Results
- 5. Time estimates & resources
- 6. References (BibTeX)

**Use this for:**
- Thesis/dissertation context
- Explaining your approach
- Time/resource estimates for faculty
- Literature comparisons

### For DL Course (April 30 Deadline)

👉 **[🎓 SUBMISSION_30APRIL.md](SUBMISSION_30APRIL.md)**

- Problem definition template
- Dataset distribution chart guide
- Architecture diagram hints
- Results presentation
- Live demo checklist
- Sample talking points

**Use this for:**
- Course presentation preparation
- Dataset visualization code
- What to highlight to professor
- Timeline for 30 April deadline

### For Setup & Requirements

👉 **[⚙️ SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md)**

- Python 3.11+ installation
- CUDA 12.1 + cuDNN setup (for NVIDIA GPU)
- Git & VS Code setup
- Docker installation (optional)
- Installation checklist
- Troubleshooting guide

**Use this for:**
- Understanding what software to install
- CUDA/GPU configuration
- Fixing installation errors

👉 **[🔧 SETUP_LOCAL.md](SETUP_LOCAL.md)**

- Detailed local venv setup
- Prerequisites
- Troubleshooting guide
- Performance tips
- FAQs

**Use this for:**
- Understanding each setup step
- Debugging installation issues
- GPU/CUDA configuration

### For Training & Resume

👉 **[🔄 RESUME_TRAINING.md](RESUME_TRAINING.md)**

- Auto-save checkpoint mechanism
- How to resume from best model
- Resume vs cold start
- GTX 1650 workflow example
- Troubleshooting resume
- List of checkpoint files

**Use this for:**
- **Closing laptop without losing progress**
- Understanding checkpoint system
- Multi-session training plan
- GTX 1650 optimization

---

## 🏗️ Configuration

### Environment Setup

- **[devbox.json](devbox.json)** — Nix-based reproducible environment
  - Contains all scripts/commands
  - Auto-installs dependencies

- **[Dockerfile](Dockerfile)** — Multi-stage Docker builds
  - CPU runtime
  - GPU runtime (CUDA 12.1)

- **[docker-compose.yml](docker-compose.yml)** — Service orchestration
  - API service
  - Frontend service
  - Training services (GPU profile)

### Model Configuration

- **[configs/config.yaml](configs/config.yaml)** — Base config
- **[configs/stage1/mae.yaml](configs/stage1/mae.yaml)** — MAE pretraining
- **[configs/stage2/finetune.yaml](configs/stage2/finetune.yaml)** — Fine-tuning
- **[configs/data/isic2019.yaml](configs/data/isic2019.yaml)** — Data paths/classes

---

## 💻 Source Code

### Models

- **[src/raresight/models/mae.py](src/raresight/models/mae.py)**
  - Masked Autoencoder (encoder + decoder)
  - Self-supervised pretraining

- **[src/raresight/models/classifier.py](src/raresight/models/classifier.py)**
  - ViT-based classifier + head
  - Layer-wise LR decay (LLRD)

### Data & Training

- **[src/raresight/data/dataset.py](src/raresight/data/dataset.py)**
  - ISIC 2019 dataset loader
  - Augmentation pipelines

- **[src/raresight/training/trainer.py](src/raresight/training/trainer.py)**
  - Generic training loop
  - Mixed precision (AMP), gradient clipping, checkpointing

- **[src/raresight/training/losses.py](src/raresight/training/losses.py)**
  - Focal loss implementation
  - Loss factory

### Evaluation

- **[src/raresight/evaluation/metrics.py](src/raresight/evaluation/metrics.py)**
  - AUC, F1, balanced accuracy, Cohen's kappa
  - Per-class metrics

### API & Frontend

- **[api/main.py](api/main.py)**
  - FastAPI endpoints
  - Image upload → predictions
  - Health checks

- **[frontend/app.py](frontend/app.py)**
  - Streamlit UI
  - 3 tabs: Diagnosis, Dataset Info, How It Works

---

## 🔧 Scripts

### Data Preparation

- **[scripts/download_data.py](scripts/download_data.py)**
  - Load ISIC 2019 metadata
  - Generate train/val/test CSVs
  - Image preprocessing

- **[scripts/preprocess.py](scripts/preprocess.py)**
  - Verify dataset setup
  - Check PyTorch + GPU availability

### Training

- **[scripts/train_stage1_pretrain.py](scripts/train_stage1_pretrain.py)**
  - MAE self-supervised pretraining
  - ~24-48 hours (GPU)

- **[scripts/train_stage2_finetune.py](scripts/train_stage2_finetune.py)**
  - Supervised fine-tuning with focal loss
  - ~4-8 hours (GPU)

### Evaluation

- **[scripts/evaluate.py](scripts/evaluate.py)**
  - Full evaluation pipeline
  - Confusion matrix, ablation study
  - Grad-CAM visualizations

---

## 📋 Workflow Files

### Development

- **[Makefile](Makefile)** — Quick commands
  - `make setup`, `make train-s1`, `make api`, etc.

- **[pyproject.toml](pyproject.toml)** — Poetry dependencies
  - PyTorch, FastAPI, Streamlit, Hydra, etc.

- **[.gitignore](.gitignore)** — Git ignore rules
  - Excludes data, checkpoints, caches, environments

### Testing

- **[tests/test_models.py](tests/test_models.py)** — Unit tests
  - Model forward pass tests
  - Shape verification

---

## 📊 Workspace Structure

```
raresight/
├── src/raresight/          # Main package
├── scripts/                # Training & evaluation
├── api/                    # FastAPI backend
├── frontend/               # Streamlit UI
├── configs/                # Hydra configs
├── tests/                  # Pytest tests
├── docs/                   # Documentation (this folder)
│
├── README.md               # Project overview
├── TRAINING_STEPS.md       # Step-by-step training
├── DEVBOX_QUICK.md         # Devbox commands
├── SETUP_LOCAL.md          # Local setup guide
├── DOCKER_RANCHER.md       # Container setup
├── QUICK_START.md          # 5-min demo
├── ACADEMIC_DOCUMENTATION.md  # Research docs
├──INDEX.md                 # THIS FILE
│
├── Dockerfile              # Docker build
├── docker-compose.yml      # Compose services
├── devbox.json             # Devbox config
├── Makefile                # Quick commands
├── pyproject.toml          # Poetry deps
├── .gitignore              # Git ignore
```

---

## 🎯 Common Tasks

### "I want to train the model"
→ [TRAINING_STEPS.md](TRAINING_STEPS.md) or [DEVBOX_QUICK.md](DEVBOX_QUICK.md)

### "I need to write a dissertation"
→ [ACADEMIC_DOCUMENTATION.md](ACADEMIC_DOCUMENTATION.md)

### "I want to use Docker"
→ [DOCKER_RANCHER.md](DOCKER_RANCHER.md)

### "I need to deploy the API"
→ [api/main.py](api/main.py) & [DOCKER_RANCHER.md](DOCKER_RANCHER.md)

### "I want to test without training"
→ [QUICK_START.md](QUICK_START.md)

### "I need to understand the code"
→ [src/raresight/](src/raresight/) & [scripts/](scripts/)

---

## 📚 Quick Reference

### Commands (with Devbox)
```bash
devbox run setup            # Install everything
devbox run prepare          # Create directories
devbox run download         # Generate CSVs
devbox run train            # Train both stages
devbox run api              # Start API
devbox run frontend         # Start Frontend
```

### Commands (without Devbox)
```bash
poetry install
python scripts/download_data.py
python scripts/train_stage1_pretrain.py
python scripts/train_stage2_finetune.py
python -m uvicorn api.main:app --reload --port 8000
streamlit run frontend/app.py --server.port 8501
```

### URLs
- **API Docs:** http://localhost:8000/docs
- **API Predictions:** http://localhost:8000/predict
- **Frontend:** http://localhost:8501
- **Rancher UI:** http://localhost:8080 (if using Rancher)

---

## ❓ FAQs

**Q: Which path should I choose?**  
A: If new to ML → use Devbox. If experienced → use manual setup. If deployment → use Docker.

**Q: How long does training take?**  
A: ~24-48 hours on GPU A100. Check [ACADEMIC_DOCUMENTATION.md](ACADEMIC_DOCUMENTATION.md) for estimates.

**Q: Do I need GPU?**  
A: No, but CPU training is ~50x slower.

**Q: What's the image-only model?**  
A: Accepts only dermoscopy images (224×224). No clinical features.

**Q: Can I use my own data?**  
A: Yes, modify [scripts/download_data.py](scripts/download_data.py) to load your dataset.

---

**Start here:** [🎁 DEVBOX_QUICK.md](DEVBOX_QUICK.md) or [📖 TRAINING_STEPS.md](TRAINING_STEPS.md)

**Last Updated:** April 2026  
**Status:** ✅ Production Ready
