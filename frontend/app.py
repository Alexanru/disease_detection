"""Streamlit frontend for RareSight."""

from __future__ import annotations

import os

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://localhost:8000")
HAM10000_LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot", "genital",
    "hand", "lower extremity", "neck", "scalp", "trunk", "unknown", "upper extremity",
]

if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "age_value" not in st.session_state:
    st.session_state["age_value"] = 45
if "sex_value" not in st.session_state:
    st.session_state["sex_value"] = "unknown"
if "localization_value" not in st.session_state:
    st.session_state["localization_value"] = "unknown"

st.set_page_config(
    page_title="RareSight",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)


def fetch_api_info() -> dict:
    try:
        health = requests.get(f"{API_URL}/health", timeout=3)
        info = requests.get(f"{API_URL}/info", timeout=3)
        if health.ok and info.ok:
            return {"online": True, "health": health.json(), "info": info.json()}
    except requests.RequestException:
        pass
    return {
        "online": False,
        "health": {"status": "offline", "model_loaded": False, "model_mode": "unknown", "checkpoint": ""},
        "info": {"requires_clinical": False, "accepted_localizations": HAM10000_LOCALIZATIONS, "model_mode": "unknown", "checkpoint": ""},
    }


api_state = fetch_api_info()
health = api_state["health"]
info = api_state["info"]
is_stage3 = info.get("model_mode") == "stage3"

with st.sidebar:
    st.markdown("## RareSight")
    st.markdown("Dermatology inference demo for the RareSight project.")
    st.markdown(f"**API status:** {'Online' if api_state['online'] else 'Offline'}")
    st.markdown(f"**Loaded mode:** {health.get('model_mode', 'unknown')}")
    checkpoint = health.get("checkpoint") or "not loaded"
    st.caption(f"Checkpoint: {checkpoint}")
    if not api_state["online"]:
        st.warning("Start the API in another terminal before using the interface.")
    elif is_stage3:
        st.info("The API is using the Stage 3 multimodal model. Clinical fields are enabled.")
    else:
        st.info("The API is using the Stage 2 image-only classifier.")

st.title("RareSight")
st.write(
    "Upload a dermoscopy image and run either the image-only classifier or the multimodal model, "
    "depending on the checkpoint loaded by the API."
)

tab1, tab2, tab3 = st.tabs(["Diagnosis", "What To Test", "How It Works"])

with tab1:
    left, right = st.columns([1, 1], gap="large")
    with left:
        cols = st.columns([1, 1])
        with cols[0]:
            reset = st.button("Reset form")
        with cols[1]:
            clear_result = st.button("Clear results")

        if reset:
            st.session_state["uploader_key"] += 1
            st.session_state["age_value"] = 45
            st.session_state["sex_value"] = "unknown"
            st.session_state["localization_value"] = "unknown"
            st.session_state.pop("last_result", None)
            st.rerun()

        if clear_result:
            st.session_state.pop("last_result", None)
            st.rerun()

        uploaded_file = st.file_uploader(
            "Choose a dermoscopy image",
            type=["jpg", "jpeg", "png"],
            help="Use dermoscopy or close-up lesion images in JPG or PNG format.",
            key=f"uploader_{st.session_state['uploader_key']}",
        )

        age = None
        sex = None
        localization = None
        if is_stage3:
            st.markdown("### Clinical metadata")
            age = st.number_input("Age", min_value=0, max_value=100, value=st.session_state["age_value"], step=1)
            sex = st.selectbox(
                "Sex",
                ["unknown", "female", "male"],
                index=["unknown", "female", "male"].index(st.session_state["sex_value"]),
            )
            localization = st.selectbox(
                "Localization",
                HAM10000_LOCALIZATIONS,
                index=HAM10000_LOCALIZATIONS.index(st.session_state["localization_value"]),
            )
            st.session_state["age_value"] = int(age)
            st.session_state["sex_value"] = sex
            st.session_state["localization_value"] = localization

        predict = st.button("Run prediction", type="primary", disabled=uploaded_file is None or not api_state["online"])

    with right:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, width="stretch", caption="Uploaded image")
        else:
            st.info("Upload an image to see the preview and run inference.")

    if predict and uploaded_file is not None:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type or "image/jpeg")}
        data = {}
        if is_stage3:
            data = {"age": str(age), "sex": sex, "localization": localization}
        try:
            response = requests.post(f"{API_URL}/predict", files=files, data=data, timeout=30)
            if not response.ok:
                st.error(f"Backend error: {response.status_code}\n{response.text}")
            else:
                result = response.json()
                st.session_state["last_result"] = result
                st.session_state["last_uploaded_name"] = uploaded_file.name
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        st.caption(f"Last prediction for: {st.session_state.get('last_uploaded_name', 'uploaded image')}")
        top_pred = result["top_prediction"]
        summary = (
            f"Diagnosis: {top_pred['class_name']}\n\n"
            f"Confidence: {top_pred['probability'] * 100:.1f}%\n\n"
            f"ICD-10: {top_pred['icd10']}"
        )
        if top_pred["is_rare"]:
            st.error(summary)
        else:
            st.success(summary)

        st.metric("Rare disease risk", f"{result['rare_disease_risk'] * 100:.1f}%")
        st.caption(f"Inference time: {result['processing_time_ms']:.0f} ms")

        st.subheader("All class probabilities")
        table_rows = [
            {
                "Class": pred["class_name"],
                "Probability": f"{pred['probability'] * 100:.2f}%",
                "Rare": "Yes" if pred["is_rare"] else "No",
            }
            for pred in result["all_predictions"]
        ]
        st.table(table_rows)
        st.info(result.get("disclaimer", ""))

with tab2:
    st.markdown(
        """
### What you should test

1. Open `http://localhost:8000/health` and confirm:
   - `status` is `ok`
   - `model_loaded` is `true`
   - `model_mode` matches the checkpoint you intended to load

2. Upload a dermoscopy image in the frontend.

3. For Stage 3, also fill the clinical fields:
   - age
   - sex
   - lesion localization

4. Confirm that the app returns:
   - one top prediction
   - a table with probabilities for all classes
   - a rare disease risk score
   - inference time

### Where to get a test image

- A lesion image from the ISIC archive
- A sample from the HAM10000 dataset already present in your local `data/raw/HAM10000` folders

If you want a quick local test, you can simply pick any `.jpg` from:

- `data/processed/isic2019/images/`
- `data/raw/HAM10000/HAM10000_images_part_1/`
- `data/raw/HAM10000/HAM10000_images_part_2/`

### What a good test looks like

- The backend does not crash
- The frontend shows a result quickly
- Repeating the same image gives stable predictions
- Stage 3 accepts clinical metadata and still returns a prediction cleanly
        """
    )

with tab3:
    st.markdown(
        """
### Model paths

- Stage 2: image-only classifier trained on ISIC 2019
- Stage 3: multimodal classifier trained on HAM10000 with age, sex, and localization

### Important note

The frontend follows whatever checkpoint the API loaded. If the API is started with a Stage 2 checkpoint,
the app behaves as image-only. If the API is started with `multimodal_best.pth`, the app enables the
clinical form and sends those values to the backend.

### Recommended run commands

Stage 2:
```bash
MODEL_CHECKPOINT=checkpoints/finetune_fast_best.pth python -m uvicorn api.main:app --reload --port 8000
python -m streamlit run frontend/app.py --server.port 8501
```

Stage 3:
```bash
MODEL_MODE=stage3 MODEL_CHECKPOINT=checkpoints/multimodal_best.pth python -m uvicorn api.main:app --reload --port 8000
python -m streamlit run frontend/app.py --server.port 8501
```
        """
    )
