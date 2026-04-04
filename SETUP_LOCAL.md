# 🔬 RareSight — Setup & Training Guide (Local venv)

**Complete guide for training on your machine with local Python venv.**

---

## Prerequisites

- **Python 3.11+** installed
- **~50GB free disk space** (for data + checkpoints + training)
- **NVIDIA GPU recommended** (training on CPU is very slow)
  - If GPU: CUDA 12.1 and cuDNN 8 installed
- **Git** (optional, for version control)

---

## Stage 1: Local Setup

### 1.1 Create Virtual Environment

```bash
# Navigate to project directory
cd d:\raresight

# Create venv
python -m venv venv

# Activate venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

**Verify activation:**
```bash
python --version      # Should be 3.11+
pip list              # Should be minimal
```

### 1.2 Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install Poetry (dependency manager)
pip install poetry==1.8.3

# Install all project dependencies
poetry install
```

**This installs:**
- PyTorch 2.3.0 (with CUDA 12.1 if GPU available)
- FastAPI, Streamlit, Hydra, scikit-learn, etc.
- Development tools (pytest, ruff, mypy)

**Time:** ~10-15 minutes (depends on connection)

**Test installation:**
```bash
python -c "import torch; import streamlit; print('✅ OK')"
```

---

## Stage 2: Prepare Data (CRITICAL!)

### 2.1 Download ISIC 2019

1. **Visit official archive:** https://challenge.isic-archive.com/data/#2019
2. **Download these files:**
   - `ISIC_2019_Training_Input.zip` (~47 GB)
   - `ISIC_2019_Training_GroundTruth.csv` (~1 MB)
   - `ISIC_2019_Training_Metadata.csv` (~2 MB)

3. **Extract to project:**
   ```bash
   cd d:\raresight\data\raw
   # Extract ISIC_2019_Training_Input.zip here
   cd d:\raresight
   ```

**Directory structure after extraction:**
```
data/
├── raw/
│   ├── ISIC_2019_Training_Input/      # 25,331 .jpg images
│   ├── ISIC_2019_Training_GroundTruth.csv
│   └── ISIC_2019_Training_Metadata.csv
├── processed/                         # Will be created by script
└── ...
```

### 2.2 Generate Train/Val/Test Splits

This script reads raw data and creates 3 CSV files with splits.

```bash
# Activate venv first
venv\Scripts\activate  # Windows

# Run preprocessing
python scripts/download_data.py
```

**Expected output:**
```
🔬 RareSight Dataset Preparation
==============================================================================
📋 Loaded metadata for 25,331 images
✅ Processed 25,331 images

📊 Class distribution:
   Melanoma                      :  4,522 ( 17.8%)
   Melanocytic Nevi             : 12,875 ( 50.8%)
   ...

🔀 Creating stratified splits...
   ✅ Created train.csv: 17,732 samples
   ✅ Created val.csv: 3,800 samples
   ✅ Created test.csv: 3,799 samples

🎉 ISIC 2019 preprocessing complete!
```

**Result: CSV files created at:**
```
data/processed/
├── train.csv          # 17,732 samples
├── val.csv            # 3,800 samples
├── test.csv           # 3,799 samples
└── images/            # 25,331 resized images
```

---

## Stage 3: Train Models

### 3.1 Stage 1 — MAE Pre-training

Self-supervised learning on unlabeled images. Learns visual representations by reconstructing masked patches.

```bash
# Activate venv
venv\Scripts\activate

# Start training
python scripts/train_stage1_pretrain.py
```

**Configuration (if needed, override via CLI):**
- `--cfg stage1.training.epochs=100` (default: 100)
- `--cfg stage1.training.batch_size=64` (default: 64)
- `--cfg hardware.device=cuda` (use GPU) or `cpu`

**Expected behavior:**
- Logs to `logs/` and `checkpoints/`
- Saves best checkpoint every epoch
- Final checkpoint: `checkpoints/mae_best.pth`

**Time estimate:**
- GPU (NVIDIA A100): 1-2 days
- GPU (RTX 3080): 3-5 days
- GPU (RTX 2080): 5-7 days
- **CPU: ~2 weeks (not recommended)**

**Monitor training:**
```bash
# In another terminal, watch for checkpoints
ls -lht checkpoints/

# Or use TensorBoard (if enabled)
tensorboard --logdir=logs
```

### 3.2 Stage 2 — Supervised Fine-tuning

Loads the MAE encoder and fine-tunes with labeled data using focal loss for rare class handling.

```bash
# Make sure Stage 1 finished (checkpoint exists)
ls checkpoints/mae_best.pth  # Should exist

# Start Stage 2
python scripts/train_stage2_finetune.py
```

**Configuration override (optional):**
- `--cfg stage2.training.epochs=50` (default: 50)
- `--cfg stage2.training.lr=5e-5` (learning rate)
- `--cfg stage2.layer_decay=0.75` (layer-wise LR decay)

**Expected behavior:**
- Loads MAE checkpoint from Stage 1
- Adds classification head
- Trains with focal loss (γ=2)
- Saves best model: `checkpoints/finetune_best.pth`

**Time estimate:**
- GPU (NVIDIA A100): 2-4 hours
- GPU (RTX 3080): 4-8 hours
- GPU (RTX 2080): 8-12 hours
- **CPU: ~1 week**

### 3.3 Running Both Stages (Convenience)

```bash
# Chain them (takes ~full day on GPU)
make train-all
# or
python scripts/train_stage1_pretrain.py && python scripts/train_stage2_finetune.py
```

---

## Stage 4: Test & Evaluate

### 4.1 Evaluate Fine-tuned Model

```bash
python scripts/evaluate.py
```

Generates:
- Per-class metrics (AUC, F1, AP)
- Confusion matrix (PNG)
- Ablation study results
- Comparison vs literature

### 4.2 Run Tests

```bash
pytest tests/ -v
```

---

## Stage 5: Deploy & Use

### 5.1 Start API Backend

```bash
# Terminal 1
venv\Scripts\activate
python -m uvicorn api.main:app --reload --port 8000
```

**API automatically loads:**
- Latest checkpoint from `checkpoints/finetune_best.pth`
- Or runs in demo mode if checkpoint missing

**API endpoints:**
- `GET  http://localhost:8000/health` — health check
- `POST http://localhost:8000/predict` — predict on image
- `GET  http://localhost:8000/docs` — Swagger UI

### 5.2 Start Frontend

```bash
# Terminal 2 (new, with venv activated)
venv\Scripts\activate
streamlit run frontend/app.py --server.port 8501
```

**Frontend:**
- Opens at `http://localhost:8501`
- Upload image → get predictions
- View dataset info & model architecture

---

## Common Commands

```bash
# Activate venv (always first!)
venv\Scripts\activate

# Training
make train-s1          # Stage 1 only
make train-s2          # Stage 2 only
make train-all         # Both stages

# Evaluation
make evaluate

# API + Frontend
make api               # Terminal 1
make frontend          # Terminal 2

# Quality checks
make test              # Run pytest
make lint              # Ruff + MyPy
make format            # Auto-format code

# Utilities
make preprocess        # Check dataset setup
make clean             # Remove caches
make help              # Show all commands
```

---

## Troubleshooting

### ❌ `python: command not found`
- Ensure Python 3.11 is installed and in PATH
- Or use full path: `C:\Python311\python.exe --version`

### ❌ `ModuleNotFoundError: torch`
- Activate venv: `venv\Scripts\activate`
- Reinstall: `pip install torch torchvision`

### ❌ Out of GPU memory during training
- Reduce batch size: `--cfg stage1.training.batch_size=32`
- Or use CPU (slower): `--cfg hardware.device=cpu`

### ❌ CUDA not available
- Check NVIDIA drivers: `nvidia-smi`
- Reinstall PyTorch with CUDA: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### ❌ Data CSV not created
- Ensure ISIC 2019 raw files exist: `data/raw/ISIC_2019_Training_Input/`
- Rerun: `python scripts/download_data.py`

### ❌ Checkpoint not found during Stage 2
- Verify Stage 1 finished: `ls checkpoints/mae_best.pth`
- If missing, restart Stage 1

### ❌ API/Frontend can't connect
- Ensure both are running (separate terminals)
- Check ports not in use: 8000, 8501
- Test: `http://localhost:8000/health`

---

## Performance Tips

### Speed Up Training
- Use **GPU** (50x faster than CPU!)
- Enable **mixed precision**: `--cfg hardware.mixed_precision=true` (default)
- Increase **num_workers**: `--cfg hardware.num_workers=8` (depends on CPU cores)

### Save Disk Space
- Reduce image size in preprocessing (default: 512×512)
- Delete old checkpoints: `rm checkpoints/mae_epoch*.pth` (keep only `*_best.pth`)
- Compress logs: `tar -czf logs.tar.gz logs/`

### Monitor Training
- **TensorBoard**: `tensorboard --logdir=logs`
- **Weights & Biases** (Wandb): Set up account and API key
- **Check filesystem**: `du -sh data/` (total data size)

---

## FAQs

**Q: Do I need GPU?**  
A: No, but it's ~50x faster. CPU training takes 1-2 weeks; GPU takes 1-2 days.

**Q: Can I use RTX 2060 / older GPU?**  
A: Yes, but may need to reduce batch size or use lower precision.

**Q: Can I stop and resume training?**  
A: Use checkpoint: Edit training script to load from `checkpoints/*_best.pth`.

**Q: How do I modify config?**  
A: Edit `configs/` YAML files or use CLI overrides: `--cfg key=value`.

**Q: What if I run out of space?**  
A: Delete `data/processed/images/` and rerun `download_data.py` (keeps CSVs, regenerates images).

---

## Next Steps

1. ✅ Setup venv (§1)
2. ✅ Download & prepare data (§2)
3. ✅ Train Stage 1 (§3.1)
4. ✅ Train Stage 2 (§3.2)
5. ✅ Evaluate (§4)
6. ✅ Deploy API + Frontend (§5)

---

**Questions?** Check [README.md](README.md) or [QUICK_START.md](QUICK_START.md).

**Need help?** Create an issue on GitHub.
