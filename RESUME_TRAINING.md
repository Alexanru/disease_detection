# 🔄 How to Resume Training (Auto-Save)

**You can now close your laptop anytime without losing progress!**

---

## **How It Works**

Your training scripts now **automatically save checkpoints** and can **resume from where you stopped**.

```
Training Session 1 (Stopped after epoch 25):
  python scripts/train_stage1_pretrain.py
  → Saves outputs/mae_epoch020.pth, mae_epoch025.pth, mae_best.pth
  → [You close laptop]

Training Session 2 (Resume from best checkpoint):
  python scripts/train_stage1_pretrain.py --resume
  → Loads mae_best.pth (from epoch 25)
  → Continues training from epoch 26
  → [You can close laptop again anytime]
```

---

## **Commands**

### **Start Training (Fresh)**
```bash
python scripts/train_stage1_pretrain.py
# or
python scripts/train_stage2_finetune.py
```

### **Resume from Best Checkpoint**
```bash
# Resumes from the best model saved so far
python scripts/train_stage1_pretrain.py --resume
python scripts/train_stage2_finetune.py --resume
```

### **Resume from Specific Checkpoint**
```bash
# Resumes from a particular epoch
python scripts/train_stage1_pretrain.py --resume-from outputs/mae_epoch050.pth
python scripts/train_stage2_finetune.py --resume-from outputs/finetune_epoch025.pth
```

---

## **What Gets Saved**

Every training creates these files in `outputs/`:

```
├── mae_epoch020.pth       # Checkpoint at epoch 20
├── mae_epoch025.pth       # Checkpoint at epoch 25
├── mae_best.pth           # Best model (lowest loss)
├── finetune_epoch010.pth  # Stage 2 checkpoint
├── finetune_best.pth      # Stage 2 best
```

**Best checkpoint** = The model with lowest validation loss (or training loss for MAE)

---

## **Typical Workflow with GTX 1650**

```bash
# DAY 1: Train Stage 1 (MAE)
# Terminal stays open, ~8-10 hours
python scripts/train_stage1_pretrain.py

# DAY 2: Continue Stage 1
# Pick up where you left off
python scripts/train_stage1_pretrain.py --resume

# DAY 2 (later): Continue Stage 1 again
python scripts/train_stage1_pretrain.py --resume

# DAY 3: Complete Stage 1, start Stage 2
python scripts/train_stage2_finetune.py

# DAY 3 (later): Continue Stage 2
python scripts/train_stage2_finetune.py --resume
```

---

## **Important Notes**

✅ **You can close laptop anytime** → Training pauses, checkpoint saved  
✅ **Run same command again** → Training resumes automatically  
✅ **All hyperparameters preserved** → Optimizer state, scheduler, LR all restored  
✅ **Safe to Ctrl+C** → Press Ctrl+C in terminal, it saves checkpoint before stopping  

❌ **Don't delete checkpoint files** → They're in `outputs/mae_best.pth` etc.  
❌ **Don't interrupt while saving** → Last epochs are most critical  

---

## **Where Are Checkpoints?**

Checkpoints save to: `{project_root}/outputs/`

```
d:\raresight\
  outputs/
    .hydra/          # Config files
    mae_best.pth     # Stage 1 best model
    mae_epoch*.pth   # All stages 1 epochs
    finetune_best.pth
    finetune_epoch*.pth
```

---

## **Troubleshooting**

### **"Resume from best checkpoint" but nothing changes**
- Check: `ls outputs/mae_best.pth` exists
- If not, you're starting fresh (first time)

### **Want to start completely fresh**
```bash
# Delete all checkpoints
rm outputs/mae*.pth outputs/finetune*.pth

# Then train normally
python scripts/train_stage1_pretrain.py
```

### **Changed config but want to continue training**
```bash
# Resume ignores config changes for saved hyperparameters
# To apply new config, delete checkpoint and restart
rm outputs/mae_best.pth
python scripts/train_stage1_pretrain.py
```

---

## **For Devbox Users**

```bash
# Start training
devbox run train-s1

# Close laptop, resume next day
devbox run train-s1     # Automatically resumes if checkpoint exists

# Manually specify checkpoint
devbox run train-s1 -- --resume-from outputs/mae_epoch050.pth
```

---

## **Questions?**

- Full training guide: [TRAINING_STEPS.md](TRAINING_STEPS.md)
- Setup guide: [SETUP_REQUIREMENTS.md](SETUP_REQUIREMENTS.md)
- All docs: [INDEX.md](INDEX.md)
