"""api/main.py — RareSight FastAPI backend.

Endpoints:
  GET  /health                — liveness probe
  GET  /info                  — model & dataset info
  POST /predict               — image (+ optional clinical) → diagnosis
  POST /predict/batch         — batch prediction
  GET  /datasets              — dataset statistics
  GET  /classes               — class descriptions
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from PIL import Image
from pydantic import BaseModel, Field

# ── App bootstrap ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="RareSight API",
    description="Early detection of rare dermatological conditions via multimodal AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global state ──────────────────────────────────────────────────────────────

_model: Any = None
_device: torch.device = torch.device("cpu")

CLASS_INFO = {
    0: {"name": "Melanoma",           "icd10": "C43",   "rare": False, "description": "Malignant skin tumor arising from melanocytes."},
    1: {"name": "Melanocytic Nevi",   "icd10": "D22",   "rare": False, "description": "Benign proliferation of melanocytes (moles)."},
    2: {"name": "Basal Cell Carcin.", "icd10": "C44.9", "rare": False, "description": "Most common skin cancer, rarely metastasizes."},
    3: {"name": "Actinic Keratosis",  "icd10": "L57.0", "rare": False, "description": "Pre-cancerous lesion from UV damage."},
    4: {"name": "Benign Keratosis",   "icd10": "L82",   "rare": False, "description": "Non-cancerous growths (seborrheic keratosis)."},
    5: {"name": "Dermatofibroma",     "icd10": "D23",   "rare": True,  "description": "Rare benign fibrous nodule."},
    6: {"name": "Vascular Lesion",    "icd10": "D18.0", "rare": True,  "description": "Rare benign vascular proliferation."},
    7: {"name": "Squamous Cell Carc.","icd10": "C44.9", "rare": False, "description": "Malignant tumor from squamous epithelial cells."},
}


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model() -> None:
    global _model, _device

    import os
    ckpt_path = os.getenv("MODEL_CHECKPOINT", "checkpoints/stage2_finetune_best.pth")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {_device}")

    if Path(ckpt_path).exists():
        import sys
        sys.path.insert(0, "src")
        from raresight.models.classifier import RareDiseaseClassifier

        _model = RareDiseaseClassifier(num_classes=8)
        state = torch.load(ckpt_path, map_location=_device)
        _model.load_state_dict(state.get("model_state_dict", state))
        _model.to(_device).eval()
        logger.success(f"Model loaded from {ckpt_path}")
    else:
        logger.warning(f"No checkpoint at {ckpt_path} — running in DEMO mode (random weights).")
        import sys
        sys.path.insert(0, "src")
        from raresight.models.classifier import RareDiseaseClassifier
        _model = RareDiseaseClassifier(num_classes=8).to(_device).eval()


# ── Request / Response schemas ────────────────────────────────────────────────

class ClassPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float
    is_rare: bool
    icd10: str

class PredictionResponse(BaseModel):
    top_prediction: ClassPrediction
    all_predictions: list[ClassPrediction]
    rare_disease_risk: float       # max prob across rare classes
    processing_time_ms: float
    model_version: str = "raresight-v0.1"
    disclaimer: str = "For research purposes only. Not a clinical diagnostic tool."

class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    """PIL Image → normalized tensor [1, 3, 224, 224]."""
    from torchvision import transforms
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(_device),
        model_loaded=_model is not None,
    )


@app.get("/classes")
async def get_classes() -> dict:
    return {"classes": CLASS_INFO}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image, got {file.content_type}")

    t0 = time.perf_counter()
    image_bytes = await file.read()
    tensor = preprocess_image(image_bytes).to(_device)

    with torch.no_grad():
        logits = _model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    elapsed_ms = (time.perf_counter() - t0) * 1000

    predictions = []
    for class_id, prob in enumerate(probs):
        info = CLASS_INFO.get(class_id, {})
        predictions.append(ClassPrediction(
            class_id=class_id,
            class_name=info.get("name", f"Class {class_id}"),
            probability=float(prob),
            is_rare=info.get("rare", False),
            icd10=info.get("icd10", ""),
        ))

    predictions.sort(key=lambda x: x.probability, reverse=True)
    rare_risk = max(p.probability for p in predictions if p.is_rare)

    return PredictionResponse(
        top_prediction=predictions[0],
        all_predictions=predictions,
        rare_disease_risk=float(rare_risk),
        processing_time_ms=round(elapsed_ms, 2),
    )


@app.get("/datasets")
async def dataset_stats() -> dict:
    """Return dataset statistics for the frontend dashboard."""
    return {
        "datasets": [
            {
                "name": "ISIC 2019",
                "total": 25331,
                "classes": {"Melanoma": 4522, "Melanocytic Nevi": 12875, "BCC": 3323, "AK": 867, "BKL": 2624, "Dermatofibroma": 239, "Vascular Lesion": 253, "SCC": 628},
                "purpose": "DL Lab — image classification",
            },
            {
                "name": "HAM10000",
                "total": 10015,
                "classes": {"akiec": 327, "bcc": 514, "bkl": 1099, "df": 115, "mel": 1113, "nv": 6705, "vasc": 142},
                "purpose": "Dissertation — multimodal (image + metadata)",
                "has_clinical": True,
            },
            {
                "name": "PAD-UFES-20",
                "total": 2298,
                "classes": {"BCC": 845, "MEL": 52, "SCC": 176, "ACK": 730, "NEV": 244, "SEK": 251},
                "purpose": "Dissertation — multimodal (image + 22 clinical features)",
                "has_clinical": True,
            },
        ]
    }
