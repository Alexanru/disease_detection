#!/usr/bin/env python3

import csv
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from PIL import Image
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────

ISIC_CLASSES = ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"]

HAM_LABEL_MAP = {
    "akiec": 0, "bcc": 1, "bkl": 2,
    "df": 3, "mel": 4, "nv": 5, "vasc": 6
}

# ─────────────────────────────────────────────────────────────
# ISIC 2019
# ─────────────────────────────────────────────────────────────

def load_isic():
    path = Path("data/raw/ISIC_2019_Training_GroundTruth.csv")
    df = pd.read_csv(path)
    df["label"] = df[ISIC_CLASSES].values.argmax(axis=1)
    df = df[df["label"] < 8]
    return df

def preprocess_isic(data_root: Path):
    out_dir = data_root / "processed/isic2019"
    out_img = out_dir / "images"

    if out_img.exists() and any(out_img.iterdir()):
        logger.info(f"✅ ISIC2019 deja procesat în {out_img}, sar peste preprocesare.")
        return

    logger.info("🔬 Preprocesare ISIC2019 ...")
    raw_img = data_root / "raw/ISIC_2019_Training_Input"
    out_img.mkdir(parents=True, exist_ok=True)

    df = load_isic()
    samples = []

    for _, row in df.iterrows():
        image_id = row["image"]
        candidates = list(raw_img.glob(f"{image_id}*.jpg"))
        if not candidates:
            logger.warning(f"⚠️ Imagine ISIC {image_id} nu găsită")
            continue

        dst = out_img / f"{image_id}.jpg"
        if not dst.exists():
            img = Image.open(candidates[0]).convert("RGB")
            img.thumbnail((512, 512))

            canvas = Image.new("RGB", (512, 512), (255, 255, 255))
            offset = ((512-img.width)//2, (512-img.height)//2)
            canvas.paste(img, offset)
            canvas.save(dst)

        samples.append({
            "image_id": image_id,
            "image_path": str(dst),
            "label": int(row["label"])
        })

    save_splits(pd.DataFrame(samples), out_dir)

# ─────────────────────────────────────────────────────────────
# HAM10000
# ─────────────────────────────────────────────────────────────

def preprocess_ham(data_root: Path):
    out_dir = data_root / "processed/ham10000"
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info(f"✅ HAM10000 deja procesat în {out_dir}, sar peste preprocesare.")
        return

    meta_file = data_root / "raw/HAM10000/HAM10000_metadata.csv"
    if not meta_file.exists():
        logger.error(f"⚠️ Nu am găsit fișierul metadata: {meta_file}")
        return

    logger.info("🧬 Preprocesare HAM10000 ...")
    meta = pd.read_csv(meta_file)

    # Mapează dx -> label
    meta["label"] = meta["dx"].map(HAM_LABEL_MAP)

    # Folderele cu imagini
    img_dirs = [
        data_root / "raw/HAM10000/HAM10000_images_part_1",
        data_root / "raw/HAM10000/HAM10000_images_part_2"
    ]

    samples = []

    for _, row in meta.iterrows():
        image_id = row["image_id"]
        # Caută imaginea în ambele foldere
        image_path = None
        for img_dir in img_dirs:
            candidate = list(img_dir.glob(f"{image_id}*.jpg"))
            if candidate:
                image_path = candidate[0]
                break

        if image_path is None:
            logger.warning(f"⚠️ Imagine HAM {image_id} nu găsită")
            continue

        samples.append({
            "image_id": image_id,
            "image_path": str(image_path),
            "label": int(row["label"])
        })

    if not samples:
        logger.error("⚠️ Nu s-au găsit imagini valide pentru HAM10000")
        return

    save_splits(pd.DataFrame(samples), out_dir)

# ─────────────────────────────────────────────────────────────
# SPLIT CSV
# ─────────────────────────────────────────────────────────────

def save_splits(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    train, temp = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    logger.success(f"{out_dir.name}: {len(df)} samples procesate")

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    data_root = Path("data")
    preprocess_isic(data_root)
    preprocess_ham(data_root)
    logger.success("🎉 ALL DATA READY!")

if __name__ == "__main__":
    main()