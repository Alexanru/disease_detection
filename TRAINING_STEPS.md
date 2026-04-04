# 🔬 RARESIGHT — TRAINING GUIDE (PAȘI EXACȚI)

**Ghid pas cu pas pentru antrenare completă. Urmează EXACT acești pași în ordine!**

---

## 📋 Condiții preliminare

✅ **Python 3.11** instalat  
✅ **~50 GB spațiu liber** (date + modele)  
✅ **GPU NVIDIA** (recomandat) sau CPU (mult mai lent)  
✅ **Proiectul RareSight** clonat local  

---

## 🚀 PAȘI DE EXECUȚIE (În Ordine)

### PASUL 1: Activează venv local

```bash
cd d:\raresight
python -m venv venv
venv\Scripts\activate
```

**Verificare:**
```bash
python --version           # Should be 3.11+
pip list | findstr poetry  # Should be empty
```

---

### PASUL 2: Instalează dependențele

```bash
pip install --upgrade pip
pip install poetry==1.8.3
poetry install
```

**Timp estimat:** 10-15 minute  
**Verific:**
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

### PASUL 3: Descarcă ISIC 2019 (MANUAL!)

**⚠️ ACEASTĂ PARTE NU SE AUTOMATIZEAZĂ — TREBUIE FĂCUT MANUAL**

1. **Deschide browser:**
   - URL: https://challenge.isic-archive.com/data/#2019

2. **Descarcă 3 fișiere:**
   - `ISIC_2019_Training_Input.zip` (~47 GB) — **Asta-i MARE!**
   - `ISIC_2019_Training_GroundTruth.csv` (~1 MB)
   - `ISIC_2019_Training_Metadata.csv` (~2 MB)

3. **Extrage în proiect:**
   ```bash
   # După ce sunt descărcate, extrage-le:
   cd d:\raresight\data\raw
   # Extrage ISIC_2019_Training_Input.zip aici
   # Copiază cele 2 CSV-uri aici
   cd d:\raresight
   ```

4. **Verific structura:**
   ```bash
   ls data/raw/
   # Ar trebui să vezi:
   # - ISIC_2019_Training_Input/     (folder cu 25,331 imagini)
   # - ISIC_2019_Training_GroundTruth.csv
   # - ISIC_2019_Training_Metadata.csv
   ```

---

### PASUL 4: Preproceseaza datele (Genereaza CSV-uri)

**Scriptul ăsta citeste datele brute și creează train.csv, val.csv, test.csv**

```bash
python scripts/download_data.py
```

**Output așteptat:**
```
🔬 RareSight Dataset Preparation
============================================================================

📋 Loaded metadata for 25,331 images
✅ Processed 25,331 images

📊 Class distribution:
   Melanoma                      :  4,522 ( 17.8%)
   Melanocytic Nevi             : 12,875 ( 50.8%)
   Basal Cell Carcinoma         :  3,323 ( 13.1%)
   ...

🔀 Creating stratified splits...
   ✅ Created train.csv: 17,732 samples
   ✅ Created val.csv: 3,800 samples
   ✅ Created test.csv: 3,799 samples

🎉 ISIC 2019 preprocessing complete!
```

**Verific că s-a créat:**
```bash
ls data/processed/
# Ar trebui să vezi: train.csv, val.csv, test.csv, images/
```

---

### PASUL 5: Antrenare ETAPA 1 (MAE Pre-training)

**⏱️ TIMP: GPU ~20-48 ore, CPU ~2 săptămâni**

```bash
python scripts/train_stage1_pretrain.py
```

**Ce se întâmplă:**
- Modelul learns să reconstruiească imagini mascate (75% patches ascunse)
- Self-supervised learning pe 17,732 imagini de antrenare
- Salvează checkpoint: `checkpoints/mae_best.pth`
- Logs în `logs/` folder

**Să stii:**
- Se afiseaza loss la fiecare epoch
- Poți **CTRL+C** pentru a opri și relua mai târziu (nu se pierde progresul datorită checkpoint-urilor)
- Checkpoint-ul "best" se actualizeaza automat

**Verific progres:**
```bash
# În alt terminal, verifică dimensiunea checkpoint-ulo
ls -lh checkpoints/
# sau direct:
du -sh checkpoints/mae_best.pth  # Ar trebui ~350MB
```

**După finalizare:**
```bash
ls checkpoints/mae_best.pth  # Ar trebui să existe
```

---

### PASUL 6: Antrenare ETAPA 2 (Fine-tuning supervizat)

**⏱️ TIMP: GPU ~4-8 ore, CPU ~1 săptămână**

```bash
python scripts/train_stage2_finetune.py
```

**Ce se întâmplă:**
- Încarcă encoder de la PASUL 5 ✅
- Adaugă classification head (8 clase)
- Antrenează cu Focal Loss (handle classe rare)
- Weighted sampling (oversampling clase rare)
- Salvează best model: `checkpoints/finetune_best.pth`

**Output așteptat:**
```
Stage 2: Supervised Fine-Tuning
...
E  1 | train_loss=2.31 | val_loss=1.95 | val_acc=0.42
E  2 | train_loss=1.87 | val_loss=1.23 | val_acc=0.65
...
E 50 | train_loss=0.42 | val_loss=0.58 | val_acc=0.85
✅ New best (val/loss=0.58) → checkpoints/finetune_best.pth
```

**După finalizare:**
```bash
ls checkpoints/finetune_best.pth  # Ar trebui să existe
```

---

### PASUL 7: Evaluare

```bash
python scripts/evaluate.py
```

**Generează:**
- Per-class metrics (AUC, F1, AP)
- Confusion matrix (PNG)
- Ablation study
- Comparison cu literature

**Output:** `outputs/evaluation/`

---

### PASUL 8: Rulează aplicația (API + Frontend)

**TERMINAL 1 — Backend API:**
```bash
# Activează venv dacă nu e deja
venv\Scripts\activate

# Pornește API
python -m uvicorn api.main:app --reload --port 8000
```

**Ar trebui să vezi:**
```
Uvicorn running on http://127.0.0.1:8000
Press CTRL+C to quit
```

**Verific API:**
- Deschide browser: http://localhost:8000/docs ✅
- Ar trebui swagger API UI

**TERMINAL 2 (NOU) — Frontend:**
```bash
venv\Scripts\activate
streamlit run frontend/app.py --server.port 8501
```

**Ar trebui să se deschidă automat:**
```
http://localhost:8501
```

---

## ✅ SUCCES!

Acum poți:
1. **Upload imagine** → **Get prediction** ✅
2. **Explore datasets** tab ✅
3. **View model architecture** ✅

---

## 📊 Checklist - Ce ar trebui să existe după toți pașii

```
✅ data/
   ├── raw/
   │   ├── ISIC_2019_Training_Input/     (25,331 imagini)
   │   ├── ISIC_2019_Training_GroundTruth.csv
   │   └── ISIC_2019_Training_Metadata.csv
   └── processed/
       ├── train.csv                      (17,732)
       ├── val.csv                        (3,800)
       ├── test.csv                       (3,799)
       └── images/                        (25,331 resized)

✅ checkpoints/
   ├── mae_best.pth                       (~350 MB)
   └── finetune_best.pth                  (~350 MB)

✅ logs/
   └── (training logs from tensorboard)

✅ outputs/evaluation/
   └── (evaluation results)

✅ venv/                                  (virtual environment)
```

---

## 🆘 Probleme & Soluții

| Problem | Cause | Solution |
|---------|-------|----------|
| `python: not found` | Python not in PATH | Install Python 3.11 + add to PATH |
| `venv: no such file` | venv not created | Run `python -m venv venv` |
| `ModuleNotFoundError` | Deps not installed | Run `poetry install` |
| `No checkpoint found` at Stage 1 finish | Stage 1 incomplete | Rerun stage 1 |
| `CUDA not available` | GPU drivers missing | Check `nvidia-smi` |
| `Out of memory` | GPU too small | Reduce batch_size in config |
| CSVs not created | Raw data missing | Check `data/raw/` structure |
| API won't start | Port 8000 in use | Use `--port 8001` instead |
| Frontend can't connect | API not running | Start API in separate terminal |

---

## ⚡ Quick Reference

```bash
# Activează venv (FÖRST DIN ORICE ALTCEVA!)
venv\Scripts\activate

# Pași de antrenare
python scripts/download_data.py     # Paso 4: Preprocess
python scripts/train_stage1_pretrain.py   # Pasul 5: Pre-train
python scripts/train_stage2_finetune.py   # Pasul 6: Fine-tune
python scripts/evaluate.py          # Pasul 7: Evaluate

# Rulare aplicație
python -m uvicorn api.main:app --reload --port 8000   # Terminal 1
streamlit run frontend/app.py --server.port 8501       # Terminal 2

# Utilități
python scripts/preprocess.py        # Check setup
make clean                          # Remove caches
```

---

## 🎓 Ce se întâmplă sub capotă

1. **Stage 1 (MAE):** Modelul invață să reconstruiasca imagini mascate
   - Encoder: ViT-Base 12 straturi
   - Decoder: 8 straturi mai ușor
   - Loss: MSE pe patch-uri mascate

2. **Stage 2 (Fine-tuning):** Modelul invață să clasifice 8 clase
   - Encoder: Încărcat din Stage 1
   - Head: Linear 768 → 8 clase
   - Loss: Focal Loss (γ=2) pentru clase rare
   - Sampling: Weighted random (oversampled rare classes)

3. **Evaluare:** Full metrics pe test set
   - Per-class AUC, F1, AP
   - Confusion matrix
   - Comparison vs literature

---

## 📚 Resurse

- **MAE Paper:** https://arxiv.org/abs/2111.06377 (He et al., 2022)
- **ViT Paper:** https://arxiv.org/abs/2010.11929 (Dosovitskiy et al., 2020)
- **Focal Loss:** https://arxiv.org/abs/1708.02002 (Lin et al., 2017)
- **ISIC Archive:** https://isic-archive.com

---

## 📞 Help

- ❓ **Config questions?** Edit `configs/stage1/mae.yaml` or `configs/stage2/finetune.yaml`
- ❓ **Code issues?** Check `SETUP_LOCAL.md` troubleshooting
- ❓ **Dataset issues?** Verify `data/` structure matches checklist above

**Happy training! 🚀**
