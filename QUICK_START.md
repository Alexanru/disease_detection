# 🚀 QUICK START — RareSight

**Get RareSight running in 5 minutes without training.**

---

## Prerequisites

- **Python 3.11+** ([download](https://www.python.org))
- **Git** (optional, for cloning)
- **1 GB free disk space** (for dependencies)
- **Windows / Mac / Linux**

---

## Option 1: With Devbox (Easiest)

### Step 1: Install Devbox

Download: https://www.jetify.com/devbox

### Step 2: Start Environment

```bash
cd d:\raresight
devbox shell
```

### Step 3: Start Backend (Terminal 1)

```bash
devbox run api
```

You should see:
```
Uvicorn running on http://127.0.0.1:8000
```

### Step 4: Start Frontend (Terminal 2)

```bash
devbox run frontend
```

Browser opens automatically. If not:
```
http://localhost:8501
```

---

## Option 2: Manual Setup (Mac / Linux / Windows)

### Step 1: Install Poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
export PATH="$HOME/.local/bin:$PATH"
```

### Step 2: Setup

```bash
poetry install
poetry shell
```

### Step 3: Start Backend (Terminal 1)

```bash
poetry run uvicorn api.main:app --reload --port 8000
```

### Step 4: Start Frontend (Terminal 2)

```bash
poetry run streamlit run frontend/app.py --server.port 8501
```

Browser opens at `http://localhost:8501`.

---

## Option 3: Docker (No local install needed)

```bash
docker compose up --build -d api frontend
```

- API: `http://localhost:8000`
- Frontend: `http://localhost:8501`

---

## ✅ You're done!

### What you can do now:

1. **Upload an image** in the frontend and get predictions ✅
2. **Explore datasets** in the Dataset Info tab ✅
3. **View API docs** at `http://localhost:8000/docs` ✅

### Demo Mode

Currently, the backend runs in **DEMO mode** (random model weights). This is safe for testing the UI.

To use a **real trained model**, place your checkpoint at:
```
checkpoints/stage2_finetune_best.pth
```

---

## 🎓 Training Your Own Model

Once you've tested the demo, train your own model:

### Step 1: Download Dataset

See [README.md](README.md#datasets) for dataset acquisition.

```bash
devbox run download  # (or download manually)
```

### Step 2: Preprocess

```bash
devbox run preprocess
```

### Step 3: Train (GPU recommended)

```bash
devbox run train-s1   # Stage 1: MAE pre-training (~1-3 days on GPU)
devbox run train-s2   # Stage 2: Fine-tuning (~2-6 hours on GPU)
```

### Step 4: Evaluate

```bash
devbox run evaluate
```

### Step 5: Update Frontend

Your new checkpoint will be loaded automatically:
```
checkpoints/stage2_finetune_best.pth
```

---

## 📋 Common Commands

```bash
# Run tests
make test

# Lint & type-check
make lint

# Format code
make format

# Clean up cache
make clean

# View all commands
make help
```

---

## ⚠️ Troubleshooting

### `poetry: command not found`
- Ensure Poetry is installed: `pip install poetry`
- Or use `python -m poetry` instead of `poetry`

### `ModuleNotFoundError: torch`
- Reinstall dependencies: `poetry install`
- Or activate venv: `poetry shell`

### Frontend can't connect to API
- Check API is running: `make api` in another terminal
- Check `http://localhost:8000/health` in browser

### Windows execution policy error
- Run in PowerShell AS ADMIN:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

---

## 🎯 Next Steps

1. **Explore the code**: Check `src/raresight/`
2. **Read the docs**: See [README.md](README.md)
3. **Read the paper**: [MAE arxiv](https://arxiv.org/abs/2111.06377)
4. **Contribute**: Make PRs with improvements!

---

**Questions?** Check [README.md](README.md) or create an issue on GitHub.
