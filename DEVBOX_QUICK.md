# 🎁 DEVBOX Quick Reference

**Auto-setup with separate commands for each step.**

---

## What is Devbox?

Devbox = automatic environment setup (installs Python 3.11, Poetry, dependencies, etc.)

**Get Devbox:** https://www.jetify.com/devbox

---

## Commands (One per Step)

### 1️⃣ **Install Everything**
```bash
devbox run setup
```
✅ Installs Poetry + all Python packages
✅ Sets up PYTHONPATH automatically

### 2️⃣ **Prepare Directories**
```bash
devbox run prepare
```
✅ Creates `data/`, `checkpoints/`, `logs/`, `outputs/`

### 3️⃣ **Prepare Data (Generate CSVs)**
```bash
devbox run download
# OR
devbox run preprocess
```
⚠️ First, download ISIC 2019 manually to `data/raw/`

### 4️⃣ **Train Stage 1 (MAE Pre-training)**
```bash
devbox run train-s1
```
⏱️ GPU: ~24 hours  
⏱️ CPU: ~2 weeks

### 5️⃣ **Train Stage 2 (Fine-tuning)**
```bash
devbox run train-s2
```
⏱️ GPU: ~4 hours  
⏱️ CPU: ~1 week

### 6️⃣ **Train Both (Chained)**
```bash
devbox run train
```
Runs: `train-s1` → `train-s2` automatically

### 7️⃣ **Evaluate**
```bash
devbox run evaluate
```
📊 Generates metrics, confusion matrix, ablation study

### 8️⃣ **Run API**
```bash
devbox run api
```
🚀 Starts FastAPI on http://localhost:8000

### 9️⃣ **Run Frontend**
```bash
devbox run frontend
```
🎨 Starts Streamlit on http://localhost:8501

### 🔟 **Run Both**
```bash
# In Terminal 1:
devbox run api

# In Terminal 2 (new):
devbox run frontend
```

---

## Full Workflow Example

```bash
# Terminal

# Step 1: Setup
devbox shell                    # Activates Devbox environment
devbox run setup                # Install dependencies (~10 min)

# Step 2: Prepare
devbox run prepare              # Create directories

# Step 3: Download data manually
# → Visit https://challenge.isic-archive.com/data/#2019
# → Extract to data/raw/
# → Then:
devbox run download             # Generate CSVs (~10 min)

# Step 4: Train
devbox run train                # Both Stage 1 + Stage 2 (~28 hours on A100)

# Step 5: Evaluate
devbox run evaluate             # Metrics + visualizations (~1 hour)

# Step 6: Run app (in 2 terminals)
# Terminal 1:
devbox run api

# Terminal 2 (new):
devbox run frontend

# Open browser: http://localhost:8501
```

---

## Other Useful Commands

```bash
# Quality checks
devbox run test                 # Run pytest
devbox run lint                 # Ruff + MyPy type check
devbox run format               # Auto-format code

# Docker commands
devbox run docker-build         # Build Docker images
devbox run docker-up            # Start Docker services
devbox run docker-down          # Stop Docker services

# Cleanup
devbox run clean                # Remove caches (__pycache__, .pytest_cache, etc.)
```

---

## Tips

### 🔄 Re-enter Environment
```bash
devbox shell         # Re-activates the environment
# OR just use:
devbox run <command>  # Automatically activates for each command
```

### 🚪 Exit Environment
```bash
exit   # or Ctrl+D
```

### 📋 See All Commands
```bash
devbox scripts
```

### 🐛 Debug
```bash
devbox info          # Show environment details
devbox doctor        # Check for issues
```

### 🔧 Custom Command
If you need to run something custom:
```bash
devbox run -- <your-command>
# Example:
devbox run -- python -c "import torch; print(torch.__version__)"
```

---

## For Academia/Dissertations

```bash
# Read academic documentation
cat ACADEMIC_DOCUMENTATION.md

# 1. Problem Statement ✅
# 2. Literature Review ✅
# 3. Methodology (MAE + Focal Loss) ✅
# 4. Time/Resource Estimates ✅
# 5. Expected Results ✅
```

---

## FAQ

**Q: What does devbox do?**  
A: Installs Python 3.11, Poetry, Docker, and all dependencies automatically.

**Q: Do I need to run `devbox shell` first?**  
A: No, `devbox run <command>` automatically activates the environment.

**Q: Can I use it without Devbox?**  
A: Yes, follow [SETUP_LOCAL.md](SETUP_LOCAL.md) for manual setup.

**Q: How long does setup take?**  
A: ~15 minutes (depends on connection + GPU drivers).

**Q: Is Devbox free?**  
A: Yes, open source.

---

**Start with:** `devbox run setup` 🚀
