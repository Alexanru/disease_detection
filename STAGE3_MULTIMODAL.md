# 🎓 Stage 3 — Multimodal Fusion Training (HAM10000)

**Transfer learning from Stage 1 checkpoint + clinical metadata fusion.**

---

## **Overview**

**Input:** 
- MAE checkpoint from Stage 1 (`mae_best.pth`)
- HAM10000 images (10K) + clinical metadata (age, sex, anatomic location)

**Output:**
- `multimodal_best.pth` — Trained multimodal classifier

**Why Stage 3?**
- Tests transfer learning from self-supervised MAE
- Combines image features with clinical metadata
- Prepares dissertation with ablation study (image-only vs multimodal)

---

## **Architecture**

```
┌─────────────────────────────────┐
│  HAM10000 Image (224×224)       │
│  ViT-B/16 encoder (from MAE)    │
│  └─ Output: [CLS] token (768D)  │
└────────────────┬────────────────┘
                 │
            [768D image repr]
                 │
      ┌──────────┴──────────┐
      │                     │
      ▼                     ▼
┌─────────────┐    ┌──────────────────┐
│  Image MLP  │    │ Clinical MLP     │
│  768 → 256  │    │ 16 → 128 → 256   │
└─────────────┘    └──────────────────┘
      │                     │
      └──────────┬──────────┘
                 │
            concat [512D]
                 │
            Fusion MLP
            512 → 512
                 │
          Classification Head
          512 → 7 classes
```

**Key features:**
- Late fusion (combine after encoding)
- Independent branches for image & clinical
- Layer-wise LR decay (LLRD) for transfer learning
- Modality dropout for robustness

---

## **Dataset — HAM10000**

| Metric | Value |
|--------|-------|
| Total images | 10,015 |
| Classes | 7 |
| Rare classes | Dermatofibroma (1.1%), Vascular Lesion (1.4%) |
| Clinical features | Age, sex, 14 anatomic locations |
| Sources | ISIC, BCN, MSK datasets |

**Anatomic locations:**
- Head/Neck, Upper extremity, Trunk, Lower extremity, Genital, Foot, Hand, etc.

---

## **Training Setup**

### **Hyperparameters**

```yaml
Epochs: 30 (with early stopping)
Batch size: 16
Optimizer: AdamW
Learning rate: 5e-5 (base)
Layer decay: 0.75 (LLRD)
Loss: Focal (γ=2.0)
Mixed precision: Yes (AMP)
```

### **Data Split**

- Train: 70% (7,010 images)
- Val: 15% (1,502 images)
- Test: 15% (1,503 images)

Stratified split preserves rare class distribution.

---

## **Running Stage 3**

### **Prerequisites**

✅ Stage 1 complete (`checkpoints/mae_best.pth` exists)  
✅ HAM10000 downloaded and extracted to `data/raw/HAM10000/`

### **Command**

```bash
# Fresh start
devbox run train-s3

# Or with custom config
devbox run train-s3 -- stage3.training.epochs=50 stage3.training.lr=1e-4

# Resume from best checkpoint
python scripts/train_stage3_multimodal.py --resume

# Resume from specific checkpoint
python scripts/train_stage3_multimodal.py --resume-from checkpoints/multimodal_epoch015.pth
```

### **Expected Output**

```
==========================================================
  RareSight — Stage 3: Multimodal Fusion Training
==========================================================

Dataset: 10,015 images | batches: 439
Multimodal model parameters: 87.3M

Stage 3 multimodal training complete ✓
Best checkpoint: checkpoints/multimodal_best.pth
Test AUC: 0.89
Test Balanced Acc: 0.82
Test Rare Recall: 0.71
```

---

## **Training Time Estimates**

| Hardware | Time | Cost |
|----------|------|------|
| **A100 GPU** | 8-12h | $0.80-1.20 |
| **RTX 3080** | 24-32h | ~$0 (personal) |
| **GTX 1650** | 3-4 days | ~$0 (personal) |
| **CPU** | 2-3 weeks | N/A |

---

## **What Gets Saved**

```
checkpoints/
├── multimodal_epoch005.pth    # Checkpoint at epoch 5
├── multimodal_epoch010.pth    # Checkpoint at epoch 10
├── multimodal_best.pth        # Best model (lowest val loss)
│
outputs/
├── .hydra/config.yaml         # Configuration snapshot
├── runs/                       # Tensorboard logs
└── wandb/                      # W&B logs (if enabled)
```

---

## **After Training — Ablation Study**

**Compare results:**

```
Comparison: ISIC 2019 vs HAM10000 Performance

| Model | Dataset | AUC | Balanced Acc | Rare Recall |
|-------|---------|-----|-------------|-------------|
| Stage 2 (image-only) | ISIC 2019 | 0.92 | 0.81 | 0.75 |
| Stage 3 (multimodal) | HAM10000  | 0.89 | 0.82 | 0.71 |

Insights:
- Image-only (Stage 2) generalizes well across datasets
- Multimodal (Stage 3) leverages clinical metadata
- Trade-off: Rare class recall vs. general accuracy
```

---

## **Troubleshooting**

### **CUDA out of memory**
```bash
# Reduce batch size
python scripts/train_stage3_multimodal.py stage3.training.batch_size=8
```

### **HAM10000 not found**
```
Error: FileNotFoundError: data/raw/HAM10000/HAM10000_metadata.csv

Fix: 
1. Download HAM10000 from Kaggle
2. Extract to data/raw/HAM10000/
3. Verify: ls data/raw/HAM10000/HAM10000_metadata.csv
```

### **MAE checkpoint not found**
```
Error: Stage 1 checkpoint missing, training from scratch

Fix:
1. Complete Stage 1 first: devbox run train-s1
2. Verify: ls checkpoints/mae_best.pth
```

---

## **For Your Dissertation**

### **What to report:**

1. **Methodology**
   - Transfer learning from MAE
   - Late fusion architecture
   - Clinical metadata integration

2. **Results**
   - AUC, balanced accuracy, rare recall
   - Per-class metrics
   - Comparison with literature

3. **Ablation study**
   - Image-only vs multimodal
   - Impact of clinical features
   - Cross-dataset generalization (ISIC 2019 → HAM10000)

4. **Analysis**
   - Which clinical features matter?
   - Failure cases?
   - When does multimodal help?

---

## **References**

- HAM10000: Tschandl et al., 2018 (https://doi.org/10.1038/sdata.2018.161)
- Multimodal fusion: Combining image and tabular data for medical diagnosis
- Transfer learning: Fine-tuning MAE encoder on new datasets

---

**Questions?** See [INDEX.md](../INDEX.md) for other guides.
