"""FastAPI backend for RareSight inference."""

from __future__ import annotations

import io
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL import Image
from pydantic import BaseModel

app = FastAPI(
    title="RareSight API",
    description="Early detection of rare dermatological conditions via multimodal AI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Any = None
_device: torch.device = torch.device("cpu")
_model_mode: str = "stage2"
_checkpoint_path: str = ""

ISIC_CLASS_INFO = {
    0: {"name": "Melanoma", "icd10": "C43", "rare": False, "description": "Malignant skin tumor arising from melanocytes."},
    1: {"name": "Melanocytic Nevi", "icd10": "D22", "rare": False, "description": "Benign proliferation of melanocytes."},
    2: {"name": "Basal Cell Carcinoma", "icd10": "C44.9", "rare": False, "description": "Common skin cancer that rarely metastasizes."},
    3: {"name": "Actinic Keratosis", "icd10": "L57.0", "rare": False, "description": "Precancerous lesion associated with sun damage."},
    4: {"name": "Benign Keratosis", "icd10": "L82", "rare": False, "description": "Non-cancerous keratotic lesion."},
    5: {"name": "Dermatofibroma", "icd10": "D23", "rare": True, "description": "Rare benign fibrous skin nodule."},
    6: {"name": "Vascular Lesion", "icd10": "D18.0", "rare": True, "description": "Rare benign vascular proliferation."},
    7: {"name": "Squamous Cell Carcinoma", "icd10": "C44.9", "rare": False, "description": "Malignant tumor of squamous epithelial cells."},
}

HAM10000_CLASS_INFO = {
    0: {"name": "Actinic Keratosis", "icd10": "L57.0", "rare": False, "description": "Precancerous lesion associated with sun damage."},
    1: {"name": "Basal Cell Carcinoma", "icd10": "C44.9", "rare": False, "description": "Common skin cancer that rarely metastasizes."},
    2: {"name": "Benign Keratosis", "icd10": "L82", "rare": False, "description": "Non-cancerous keratotic lesion."},
    3: {"name": "Dermatofibroma", "icd10": "D23", "rare": True, "description": "Rare benign fibrous skin nodule."},
    4: {"name": "Melanoma", "icd10": "C43", "rare": False, "description": "Malignant skin tumor arising from melanocytes."},
    5: {"name": "Melanocytic Nevi", "icd10": "D22", "rare": False, "description": "Benign proliferation of melanocytes."},
    6: {"name": "Vascular Lesion", "icd10": "D18.0", "rare": True, "description": "Rare benign vascular proliferation."},
}

HAM10000_LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity",
]


class ClassPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float
    is_rare: bool
    icd10: str


class PredictionResponse(BaseModel):
    top_prediction: ClassPrediction
    all_predictions: list[ClassPrediction]
    rare_disease_risk: float
    processing_time_ms: float
    model_version: str = "raresight-v0.1"
    disclaimer: str = "For research purposes only. Not a clinical diagnostic tool."


class HealthResponse(BaseModel):
    status: str
    device: str
    model_loaded: bool
    model_mode: str
    checkpoint: str


class InfoResponse(BaseModel):
    model_mode: str
    checkpoint: str
    requires_clinical: bool
    accepted_localizations: list[str]


def _class_info_for_mode(mode: str) -> dict[int, dict[str, Any]]:
    return HAM10000_CLASS_INFO if mode == "stage3" else ISIC_CLASS_INFO


def _infer_model_mode(state: dict[str, Any]) -> str:
    if any(k.startswith("clinical_encoder.") for k in state) and any(k.startswith("fusion.") for k in state):
        return "stage3"
    return "stage2"


def _infer_encoder_cfg(state: dict[str, Any], prefix: str) -> dict[str, int]:
    embed_dim = int(state[f"{prefix}cls_token"].shape[-1])
    depth = len({
        int(key[len(prefix + "blocks."):].split(".")[0])
        for key in state
        if key.startswith(f"{prefix}blocks.") and key.endswith("norm1.weight")
    })
    if embed_dim == 512:
        num_heads = 8
    elif embed_dim == 768:
        num_heads = 12
    elif embed_dim == 192:
        num_heads = 3
    else:
        raise ValueError(f"Unsupported encoder embedding size: {embed_dim}")
    return {
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "patch_size": 16,
        "img_size": 224,
    }


def _infer_stage3_num_classes(state: dict[str, Any]) -> int:
    """Infer class count from the last Linear in fusion head, regardless of index."""
    fusion_linear = []
    for key, value in state.items():
        if not key.startswith("fusion.") or not key.endswith(".weight"):
            continue
        if getattr(value, "ndim", 0) != 2:
            continue
        idx_str = key.split(".")[1]
        if idx_str.isdigit():
            fusion_linear.append((int(idx_str), int(value.shape[0])))
    if not fusion_linear:
        raise ValueError("Could not infer Stage 3 class count from fusion head weights.")
    fusion_linear.sort(key=lambda x: x[0])
    return fusion_linear[-1][1]


def _default_checkpoint() -> str:
    candidates = [
        "checkpoints/finetune_fast_best.pth",
        "checkpoints/finetune_best.pth",
        "checkpoints/multimodal_best.pth",
        "checkpoints/stage2_finetune_best.pth",
    ]
    return next((path for path in candidates if Path(path).exists()), candidates[0])


def _build_clinical_tensor(age: float | None, sex: str | None, localization: str | None) -> torch.Tensor:
    age_norm = 0.5 if age is None else max(0.0, min(float(age), 100.0)) / 100.0
    sex_value = (sex or "unknown").strip().lower()
    sex_enc = {"female": 0.0, "male": 1.0}.get(sex_value, 0.5)
    loc_value = (localization or "unknown").strip().lower()
    loc_vector = [1.0 if loc_value == loc else 0.0 for loc in HAM10000_LOCALIZATIONS]
    if sum(loc_vector) == 0:
        loc_vector[HAM10000_LOCALIZATIONS.index("unknown")] = 1.0
    return torch.tensor([age_norm, sex_enc, *loc_vector], dtype=torch.float32).unsqueeze(0)


def preprocess_image(image_bytes: bytes) -> torch.Tensor:
    from torchvision import transforms

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


@app.on_event("startup")
async def load_model() -> None:
    global _model, _device, _model_mode, _checkpoint_path

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _checkpoint_path = os.getenv("MODEL_CHECKPOINT", _default_checkpoint())
    logger.info(f"Device: {_device}")

    if not Path(_checkpoint_path).exists():
        logger.warning(f"No checkpoint at {_checkpoint_path}; running in demo mode with random weights.")
        import sys

        sys.path.insert(0, "src")
        from raresight.models.classifier import RareDiseaseClassifier

        _model = RareDiseaseClassifier(num_classes=8).to(_device).eval()
        _model_mode = "stage2"
        return

    import sys

    sys.path.insert(0, "src")
    raw_state = torch.load(_checkpoint_path, map_location=_device)
    state = raw_state.get("model_state_dict", raw_state)
    env_mode = os.getenv("MODEL_MODE", "").strip().lower()
    _model_mode = env_mode if env_mode in {"stage2", "stage3"} else _infer_model_mode(state)

    if _model_mode == "stage3":
        from raresight.models.multimodal import MultimodalFusionModel

        encoder_cfg = _infer_encoder_cfg(state, "image_encoder.")
        clinical_dim = int(state["clinical_encoder.net.0.weight"].shape[1])
        num_classes = _infer_stage3_num_classes(state)
        _model = MultimodalFusionModel(
            num_classes=num_classes,
            clinical_dim=clinical_dim,
            encoder_cfg=encoder_cfg,
        )
    else:
        from raresight.models.classifier import RareDiseaseClassifier

        encoder_cfg = _infer_encoder_cfg(state, "encoder.")
        num_classes = int(state["head.2.weight"].shape[0])
        _model = RareDiseaseClassifier(num_classes=num_classes, encoder_cfg=encoder_cfg)

    _model.load_state_dict(state)
    _model.to(_device).eval()
    logger.success(f"Model loaded from {_checkpoint_path} in {_model_mode} mode")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=str(_device),
        model_loaded=_model is not None,
        model_mode=_model_mode,
        checkpoint=_checkpoint_path,
    )


@app.get("/info", response_model=InfoResponse)
async def info() -> InfoResponse:
    return InfoResponse(
        model_mode=_model_mode,
        checkpoint=_checkpoint_path,
        requires_clinical=_model_mode == "stage3",
        accepted_localizations=HAM10000_LOCALIZATIONS if _model_mode == "stage3" else [],
    )


@app.get("/classes")
async def get_classes() -> dict:
    return {"classes": _class_info_for_mode(_model_mode), "model_mode": _model_mode}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    age: float | None = Form(default=None),
    sex: str | None = Form(default=None),
    localization: str | None = Form(default=None),
) -> PredictionResponse:
    if _model is None:
        raise HTTPException(503, "Model not loaded")
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, f"Expected image, got {file.content_type}")

    started = time.perf_counter()
    image_bytes = await file.read()
    image = preprocess_image(image_bytes).to(_device)

    with torch.no_grad():
        if _model_mode == "stage3":
            clinical = _build_clinical_tensor(age, sex, localization).to(_device)
            logits = _model(image, clinical)
        else:
            logits = _model(image)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().tolist()

    class_info = _class_info_for_mode(_model_mode)
    predictions = [
        ClassPrediction(
            class_id=class_id,
            class_name=class_info.get(class_id, {}).get("name", f"Class {class_id}"),
            probability=float(prob),
            is_rare=class_info.get(class_id, {}).get("rare", False),
            icd10=class_info.get(class_id, {}).get("icd10", ""),
        )
        for class_id, prob in enumerate(probabilities)
    ]
    predictions.sort(key=lambda item: item.probability, reverse=True)
    rare_candidates = [item.probability for item in predictions if item.is_rare]
    rare_risk = max(rare_candidates) if rare_candidates else 0.0

    return PredictionResponse(
        top_prediction=predictions[0],
        all_predictions=predictions,
        rare_disease_risk=float(rare_risk),
        processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.get("/datasets")
async def dataset_stats() -> dict:
    return {
        "datasets": [
            {
                "name": "ISIC 2019",
                "total": 25331,
                "classes": {"Melanoma": 4522, "Melanocytic Nevi": 12875, "BCC": 3323, "AK": 867, "BKL": 2624, "Dermatofibroma": 239, "Vascular Lesion": 253, "SCC": 628},
                "purpose": "Image classification",
            },
            {
                "name": "HAM10000",
                "total": 10015,
                "classes": {"Actinic Keratosis": 327, "Basal Cell Carcinoma": 514, "Benign Keratosis": 1099, "Dermatofibroma": 115, "Melanoma": 1113, "Melanocytic Nevi": 6705, "Vascular Lesion": 142},
                "purpose": "Multimodal image plus metadata",
                "has_clinical": True,
            },
        ]
    }
