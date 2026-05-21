# DermoDetection

DermoDetection is a dermatology AI project for skin lesion prediction with image-only and multimodal inference.
It supports:

- Stage 1: self-supervised MAE pretraining
- Stage 2: supervised image classifier (ISIC 2019)
- Multimodal fusion: image + clinical metadata model (HAM10000)
- FastAPI backend + Streamlit frontend for inference

---

## Table of Contents

- [About the App](#about-the-app)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Quick Start (Devbox)](#quick-start-devbox)
- [Dataset Setup](#dataset-setup)
- [Training Pipelines](#training-pipelines)
- [Run the Application](#run-the-application)
- [How to Test the App](#how-to-test-the-app)
- [Save Trained Models (GitHub Release)](#save-trained-models-github-release)
- [Restore on a New Machine](#restore-on-a-new-machine)
- [Troubleshooting](#troubleshooting)

---

## About the App

DermoDetection predicts dermatology classes from an uploaded lesion image.

- In **image-only mode**, inference is image-only.
- In **multimodal mode**, inference uses image + clinical fields:
  - age
  - sex
  - lesion localization

The frontend automatically adapts to the model mode loaded by the API.

---

## Project Structure

- `src/dermodetection/`: models, training, datasets, metrics
- `scripts/`: Stage 1 / Stage 2 / multimodal training + evaluation scripts
- `configs/`: Hydra config files
- `api/`: FastAPI backend (`api/main.py`)
- `frontend/`: Streamlit app (`frontend/app.py`)
- `checkpoints/`: trained model files (`.pth`) (local, ignored by git)
- `data/`: raw and processed datasets (local, ignored by git)

---

## Architecture

DermoDetection is organized into clear functional layers:

- `src/dermodetection/`
  - `models/`: encoder, classifier, and multimodal fusion models
  - `training/`: training loop, optimization, and loss utilities
  - `evaluation/`: metrics and evaluation helpers
  - `data/`: dataset loading, transforms, and multimodal dataset support
- `scripts/`: training, evaluation, and preprocessing entry points
- `api/`: inference service exposing `/health`, `/info`, and `/predict`
- `frontend/`: Streamlit inference UI
- `configs/`: Hydra configuration for fast and full training profiles

---

## Requirements

- Python 3.11
- Poetry
- Optional: Devbox
- Optional but recommended: CUDA GPU

If Devbox is unstable on your machine (Nix issues), use the local venv flow.

---

## Quick Start (Devbox)

### 1) Enter devbox shell

```bash
devbox shell
```

### 2) Install dependencies

```bash
devbox run setup
```

### 3) Prepare folders

```bash
devbox run prepare
```

This creates local folders like `data/`, `checkpoints/`, `logs/`, `outputs/`.

---

## Dataset Setup

Download datasets manually, then place files in the exact paths below.

### ISIC 2019

Source:
- https://challenge.isic-archive.com/data/#2019

Place files:
- images under `data/raw/ISIC_2019_Training_Input/`
- ground truth CSV as `data/raw/ISIC_2019_Training_GroundTruth.csv`

### HAM10000

Source:
- https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?resource=download

Place files:
- `data/raw/HAM10000/HAM10000_metadata.csv`
- `data/raw/HAM10000/HAM10000_images_part_1/`
- `data/raw/HAM10000/HAM10000_images_part_2/`

### Build processed splits

```bash
python scripts/download_data.py
```

Expected outputs:

- `data/processed/isic2019/{train.csv,val.csv,test.csv,images/}`
- `data/processed/ham10000/{train.csv,val.csv,test.csv}`

---

## Training Pipelines

### Fast profile (recommended for local runs)

### Stage 1

```bash
python scripts/train_stage1_pretrain.py stage1=mae_fast
```

### Image-only fine-tuning

```bash
python scripts/train_stage2_finetune.py stage1=mae_fast stage2=finetune_fast
```

### Multimodal fusion

```bash
python scripts/train_stage3_multimodal.py stage1=mae_fast
```

### Full profile (longer training)

```bash
python scripts/train_stage1_pretrain.py
python scripts/train_stage2_finetune.py
python scripts/train_stage3_multimodal.py
```

### Resume examples

```bash
python scripts/train_stage1_pretrain.py --resume stage1=mae_fast
python scripts/train_stage2_finetune.py --resume stage1=mae_fast stage2=finetune_fast
python scripts/train_stage3_multimodal.py --resume stage1=mae_fast
```

---

## Run the Application

Use **two terminals**.

### Image-only model

Terminal 1 (API):

```bash
MODEL_CHECKPOINT=checkpoints/finetune_fast_best.pth python -m uvicorn api.main:app --port 8000
```

Terminal 2 (frontend):

```bash
python -m streamlit run frontend/app.py --server.port 8501 --server.fileWatcherType none
```

### Multimodal model

Terminal 1 (API):

```bash
MODEL_MODE=stage3 MODEL_CHECKPOINT=checkpoints/multimodal_best.pth python -m uvicorn api.main:app --port 8000
```

Terminal 2 (frontend):

```bash
python -m streamlit run frontend/app.py --server.port 8501 --server.fileWatcherType none
```

Open:

- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/info`
- `http://127.0.0.1:8501`

---

## How to Test the App

### API sanity checks

1. Open `http://127.0.0.1:8000/health`
2. Confirm:
   - `status: ok`
   - `model_loaded: true`
   - `model_mode`: `stage2` for image-only inference or `stage3` for multimodal inference

### UI inference test

1. Upload a lesion image (`.jpg`/`.png`)
2. For multimodal mode, fill:
   - age
   - sex
   - localization
3. Click `Run prediction`
4. Confirm results show:
   - top class
   - confidence
   - rare disease risk
   - per-class probabilities

Good test images:

- `data/processed/isic2019/images/*.jpg`
- `data/raw/HAM10000/HAM10000_images_part_1/*.jpg`
- `data/raw/HAM10000/HAM10000_images_part_2/*.jpg`

---

## Save Trained Models (GitHub Release)

`checkpoints/` is ignored by git, so trained models are **not** pushed with normal commits.

Recommended: upload selected checkpoints to a GitHub Release.

Suggested files:

- `checkpoints/finetune_fast_best.pth`
- `checkpoints/multimodal_best.pth`
- optional: `checkpoints/mae_fast_best.pth`

### Steps

1. GitHub repo -> `Releases` -> `Create a new release`
2. Tag: for example `v1-models`
3. Title: `DermoDetection trained checkpoints`
4. Attach `.pth` files as release assets
5. Publish

Recommended release description:

```text
DermoDetection trained checkpoints for reproducible inference.

Files:
- finetune_fast_best.pth: image-only model (ISIC 2019)
- multimodal_best.pth: multimodal model (HAM10000)
- mae_fast_best.pth (optional): Stage 1 MAE encoder checkpoint

Usage:
- Stage 2 API:
  MODEL_CHECKPOINT=checkpoints/finetune_fast_best.pth python -m uvicorn api.main:app --port 8000
- Multimodal API:
  MODEL_MODE=stage3 MODEL_CHECKPOINT=checkpoints/multimodal_best.pth python -m uvicorn api.main:app --port 8000
```

---

## Restore on a New Machine

1. Clone repo
2. Setup environment (`devbox` or local venv)
3. Download release assets (`.pth`)
4. Put them in `checkpoints/`
5. Run API + frontend commands from this README

No retraining is needed if checkpoints are available.

---

## Troubleshooting

- `gio: ... Operation not supported` in WSL:
  - harmless; open browser URL manually.
- Streamlit page reload instability:
  - run with `--server.fileWatcherType none`.
- If frontend shows offline:
  - verify API is running and `/health` returns `model_loaded: true`.
- If multimodal API fails:
  - verify `MODEL_MODE=stage3`
  - verify `MODEL_CHECKPOINT=checkpoints/multimodal_best.pth`
  - verify checkpoint file exists.
