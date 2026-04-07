# RareSight

RareSight is a dermatology research project for rare skin lesion detection. It combines self-supervised pretraining, supervised fine-tuning, a FastAPI backend, and a Streamlit frontend.

The repository currently supports two practical workflows:

1. A fast local workflow meant for limited hardware and faster iteration.
2. A larger full workflow meant for longer training runs and stronger results.

## What The Project Does

- Trains a masked autoencoder on dermoscopy images in Stage 1.
- Fine-tunes a classifier on ISIC 2019 in Stage 2.
- Trains a multimodal research model on HAM10000 in Stage 3.
- Serves predictions through a FastAPI API and a Streamlit UI.

## Current Practical Status

- Stage 1 and Stage 2 form the main end-to-end application path.
- The API and Streamlit app are image-based and use a Stage 2 checkpoint by default.
- Stage 3 is useful for dissertation experiments, ablation studies, and multimodal analysis, but it is not yet wired into the default app flow.

## Repository Layout

- `src/raresight/`: core models, datasets, training utilities, evaluation helpers
- `scripts/`: training and evaluation entry points
- `configs/`: Hydra configuration files for data and training profiles
- `api/`: FastAPI inference service
- `frontend/`: Streamlit interface
- `tests/`: basic model and augmentation tests

## Data

The project uses two datasets in practice:

- `ISIC 2019`: image classification with 8 classes, used in Stage 1 and Stage 2
- `HAM10000`: image plus metadata, used in Stage 3

Expected local layout:

```text
data/
  raw/
    ISIC_2019_Training_Input/
    ISIC_2019_Training_GroundTruth.csv
    HAM10000/
      HAM10000_metadata.csv
      HAM10000_images_part_1/
      HAM10000_images_part_2/
  processed/
    isic2019/
      train.csv
      val.csv
      test.csv
      images/
    ham10000/
      train.csv
      val.csv
      test.csv
```

`scripts/download_data.py` prepares the processed CSV splits from the raw datasets already present locally.

## Environment Setup

### Option A: local virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry==1.8.3
poetry install
```

### Option B: Devbox

```bash
devbox shell
devbox run setup
```

If Devbox fails because of local Nix issues, use the local virtual environment instead.

## Training Profiles

### Fast profile

Use this profile when you want a working system quickly on modest hardware.

- Stage 1: `configs/stage1/mae_fast.yaml`
- Stage 2: `configs/stage2/finetune_fast.yaml`
- Stage 3: run Stage 3 with `stage1=mae_fast`

Commands:

```bash
python scripts/train_stage1_pretrain.py stage1=mae_fast
python scripts/train_stage2_finetune.py stage1=mae_fast stage2=finetune_fast
python scripts/train_stage3_multimodal.py stage1=mae_fast
```

Typical checkpoint outputs:

- `checkpoints/mae_fast_best.pth`
- `checkpoints/finetune_fast_best.pth`
- `checkpoints/multimodal_best.pth`

### Full profile

Use this profile when training time is less constrained and you want the larger configuration.

Commands:

```bash
python scripts/train_stage1_pretrain.py
python scripts/train_stage2_finetune.py
python scripts/train_stage3_multimodal.py
```

Typical checkpoint outputs:

- `checkpoints/mae_best.pth`
- `checkpoints/finetune_best.pth`
- `checkpoints/multimodal_best.pth`

## Resuming Training

All three training scripts support resume mode.

Examples:

```bash
python scripts/train_stage1_pretrain.py --resume stage1=mae_fast
python scripts/train_stage2_finetune.py --resume stage1=mae_fast stage2=finetune_fast
python scripts/train_stage3_multimodal.py --resume stage1=mae_fast
```

You can also resume from an explicit checkpoint path:

```bash
python scripts/train_stage1_pretrain.py --resume-from checkpoints/mae_fast_epoch010.pth stage1=mae_fast
```

## Running The Application

The default application path is Stage 2 inference.

### Start the API

If you trained the fast Stage 2 model:

```bash
MODEL_CHECKPOINT=checkpoints/finetune_fast_best.pth python -m uvicorn api.main:app --reload --port 8000
```

If you trained the full Stage 2 model:

```bash
MODEL_CHECKPOINT=checkpoints/finetune_best.pth python -m uvicorn api.main:app --reload --port 8000
```

### Start the Streamlit frontend

In a second terminal:

```bash
python -m streamlit run frontend/app.py --server.port 8501
```

Then open:

- API docs: `http://localhost:8000/docs`
- Frontend: `http://localhost:8501`

## How To Tell If The App Works

### Basic health checks

Check the API:

```bash
curl http://localhost:8000/health
```

Expected result:

- `status` should be `ok`
- `model_loaded` should be `true`

### Prediction check

Upload a dermoscopy image in the frontend or use the API docs page.

What a good basic result looks like:

- the request succeeds without backend errors
- you get one top prediction and a full class probability list
- inference time is reported
- the app does not say it is running in random demo mode

### Sanity-checking the predictions

For a quick manual sanity check:

- similar-looking benign lesions should not jump randomly between all classes
- probabilities should sum visually to a plausible distribution rather than looking flat every time
- repeated calls on the same image should give stable predictions
- the API should behave the same whether you test from Swagger or from the frontend

### Limitations

- the current frontend uses the image classifier path
- Stage 3 is trained and evaluated separately and is not the default inference path in the app
- this project is for research and dissertation use, not clinical deployment

## Evaluation

After training, you can run:

```bash
python scripts/evaluate.py
```

This script is intended for the multimodal evaluation workflow and may require adjusting the checkpoint path depending on which profile you trained.

## Important Config Files

- `configs/config.yaml`: base config for Stage 1 and Stage 2
- `configs/config_stage3.yaml`: base config for Stage 3
- `configs/data/isic2019.yaml`: ISIC 2019 dataset config
- `configs/data/ham10000.yaml`: HAM10000 dataset config
- `configs/stage1/mae.yaml`: full Stage 1 profile
- `configs/stage1/mae_fast.yaml`: fast Stage 1 profile
- `configs/stage2/finetune.yaml`: full Stage 2 profile
- `configs/stage2/finetune_fast.yaml`: fast Stage 2 profile

## Suggested Order Of Work

If you want the fastest path to a working demo:

1. Train Stage 1 fast.
2. Train Stage 2 fast.
3. Run the API and frontend with the Stage 2 fast checkpoint.
4. Train Stage 3 only after the main app path is already working.

If you want the larger experimental version:

1. Train full Stage 1.
2. Train full Stage 2.
3. Run the app with the full Stage 2 checkpoint.
4. Train Stage 3 for multimodal dissertation experiments.

## Notes For GitHub Readers

This project is designed as a research-oriented dermatology pipeline, not just a demo UI. The main value lies in:

- the staged training design
- handling rare classes in a highly imbalanced dataset
- supporting both image-only and multimodal experimentation
- exposing the trained classifier through a simple application layer
