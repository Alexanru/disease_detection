"""Streamlit frontend for DermoDetection."""

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
    st.session_state["age_value"] = ""
if "sex_value" not in st.session_state:
    st.session_state["sex_value"] = "unknown"
if "localization_value" not in st.session_state:
    st.session_state["localization_value"] = "unknown"

st.set_page_config(
    page_title="DermoDetection",
    page_icon="D",
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
is_multimodal = info.get("model_mode") == "stage3"
friendly_mode = "Multimodal" if is_multimodal else "Image-only"

with st.sidebar:
    st.markdown("## DermoDetection")
    st.markdown("Dermatology inference demo for the DermoDetection project.")
    st.markdown(f"**API status:** {'Online' if api_state['online'] else 'Offline'}")
    st.markdown(f"**Loaded mode:** {friendly_mode}")
    checkpoint = health.get("checkpoint") or "not loaded"
    st.caption(f"Checkpoint: {checkpoint}")
    if not api_state["online"]:
        st.warning("Start the API in another terminal before using the interface.")
    elif is_multimodal:
        st.info("The API is using the multimodal model with clinical fields.")
    else:
        st.info("The API is using the image-only classifier.")

st.title("DermoDetection")
st.write(
    "Upload a dermoscopy image and run either the image-only classifier or the multimodal model, "
    "depending on the checkpoint loaded by the API."
)

left, right = st.columns([1, 1], gap="large")

with left:
    cols = st.columns([1, 1])
    with cols[0]:
        reset = st.button("Reset form")
    with cols[1]:
        clear_result = st.button("Clear results")

    if reset:
        st.session_state["uploader_key"] += 1
        st.session_state["age_value"] = ""
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
    if is_multimodal:
        st.markdown("### Clinical metadata")
        age_input = st.text_input(
            "Age",
            value=st.session_state["age_value"],
            help="Enter age in years or leave blank if unknown.",
        )
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
        st.session_state["age_value"] = age_input.strip()
        st.session_state["sex_value"] = sex
        st.session_state["localization_value"] = localization
        if age_input.strip() != "":
            try:
                age = int(age_input.strip())
            except ValueError:
                age = None
        else:
            age = None

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
    if is_multimodal:
        if age is not None:
            data["age"] = str(age)
        data["sex"] = sex
        data["localization"] = localization
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
