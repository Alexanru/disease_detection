"""frontend/app.py — RareSight Streamlit frontend.

A demo UI for the RareSight multimodal dermatology classification system.
Connect to the FastAPI backend (api/main.py) for predictions.

Run with: streamlit run frontend/app.py
"""

import io
import os
from pathlib import Path

import requests
import streamlit as st
from PIL import Image

# ── Configuration ────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

st.set_page_config(
    page_title="🔬 RareSight — Rare Dermatology Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    ## 🔬 **RareSight**
    Early Detection of Rare Dermatological Conditions via AI

    ---
    
    ### 📊 About
    - **Datasets**: ISIC 2019, HAM10000, PAD-UFES-20
    - **Model**: ViT-based classifier (MAE pre-trained)
    - **Task**: Binary rare-disease detection + multi-class classification
    
    ### ⚠️ Disclaimer
    **For research purposes only.**  
    Not a clinical diagnostic tool.
    """)

    api_status = "❌ Offline"
    try:
        resp = requests.get(f"{API_URL}/health", timeout=2)
        if resp.status_code == 200:
            api_status = "✅ Online"
    except:
        api_status = "❌ Offline"

    st.markdown(f"**API Status**: {api_status}")
    if api_status == "❌ Offline":
        st.warning("Backend not responding. Ensure `devbox run api` is running on another terminal.")

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("# 🔬 RareSight")
st.markdown("""
**Early Detection of Rare Dermatological Conditions**

Upload a dermoscopy image to detect rare skin conditions (Dermatofibroma, Vascular Lesion) 
and classify 8 distinct dermatological diagnoses.
""")

st.divider()

# ── Tab layout ───────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📸 Diagnosis", "📊 Dataset Info", "ℹ️ How It Works"])

# ──────────────────────────────────────────────────────────────────────────────
# TAB 1: DIAGNOSIS
# ──────────────────────────────────────────────────────────────────────────────

with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dermoscopy image (JPG, PNG):",
            type=["jpg", "jpeg", "png"],
            help="Recommended: 224×224 or larger dermoscopy images"
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Uploaded image")

            predict_btn = st.button("🔍 Predict", key="predict", type="primary")

            if predict_btn:
                try:
                    # Send to API
                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": uploaded_file.getvalue()},
                        timeout=10,
                    )

                    if response.status_code == 200:
                        result = response.json()

                        with col2:
                            st.subheader("🎯 Results")

                            # Top prediction
                            top_pred = result["top_prediction"]
                            confidence = top_pred["probability"]
                            class_name = top_pred["class_name"]
                            is_rare = top_pred["is_rare"]

                            if is_rare:
                                st.error(
                                    f"⚠️ **RARE DISEASE DETECTED**\n\n"
                                    f"**Diagnosis**: {class_name}\n"
                                    f"**Confidence**: {confidence * 100:.1f}%\n"
                                    f"**ICD-10**: {top_pred['icd10']}"
                                )
                            else:
                                st.success(
                                    f"✅ **Diagnosis**: {class_name}\n\n"
                                    f"**Confidence**: {confidence * 100:.1f}%\n"
                                    f"**ICD-10**: {top_pred['icd10']}"
                                )

                            # Rare disease risk
                            rare_risk = result["rare_disease_risk"]
                            st.metric(
                                "Rare Disease Risk",
                                f"{rare_risk * 100:.1f}%",
                                help="Probability that the lesion is rare"
                            )

                            # Processing time
                            st.caption(f"⏱️ Inference time: {result['processing_time_ms']:.0f}ms")

                        # All predictions table
                        st.divider()
                        st.subheader("📋 All Class Probabilities")

                        all_preds = result["all_predictions"]
                        predictions_df = [
                            {
                                "Class": p["class_name"],
                                "Probability": f"{p['probability'] * 100:.2f}%",
                                "Rare": "⚠️ Yes" if p["is_rare"] else "—",
                            }
                            for p in sorted(all_preds, key=lambda x: x["probability"], reverse=True)
                        ]

                        st.table(predictions_df)

                        # Disclaimer
                        st.info(result.get("disclaimer", ""))

                    else:
                        st.error(f"Backend error: {response.status_code}\n{response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(
                        "❌ **Cannot connect to backend.**\n\n"
                        "Start the API server:\n"
                        "\`devbox run api\` (in another terminal)"
                    )
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        else:
            with col2:
                st.info("👈 Upload an image to get started.")

# ──────────────────────────────────────────────────────────────────────────────
# TAB 2: DATASET INFO
# ──────────────────────────────────────────────────────────────────────────────

with tab2:
    st.subheader("📊 Datasets Used")

    st.markdown("""
    | Dataset | Samples | Purpose | Clinical Features |
    |---|---|---|---|
    | **ISIC 2019** | 25,331 | Image classification (8 classes) | None |
    | **HAM10000** | 10,015 | Multimodal (image + metadata) | Age, sex, localization |
    | **PAD-UFES-20** | 2,298 | Multimodal (image + 22 features) | Smoke, drink, itch, diameter, … |

    ### Rare Classes (Focus Areas)
    - **Dermatofibroma (DF)**: Benign fibrous nodule (rare, often misdiagnosed)
    - **Vascular Lesion (VASC)**: Benign vascular proliferation (rare, similar to melanoma)
    """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Class Distribution (ISIC 2019)")
        st.markdown("""
        - Melanoma: 4,522 (17.8%)
        - Melanocytic Nevi: 12,875 (50.8%)
        - Benign Keratosis: 2,624 (10.4%)
        - BCC: 3,323 (13.1%)
        - Actinic Keratosis: 867 (3.4%)
        - SCC: 628 (2.5%)
        - **Vascular Lesion: 253 (1.0%)** ⚠️ RARE
        - **Dermatofibroma: 239 (0.9%)** ⚠️ RARE
        """)

    with col2:
        st.markdown("### Class Imbalance Challenge")
        st.markdown("""
        Rare classes represent <2% of data:
        - ❌ Standard classifiers ignore them
        - ✅ Focal loss helps
        - ✅ Weighted sampling helps
        - ✅ Two-stage pre-training helps
        
        **Our approach**: MAE pre-training + focal loss + weighted sampling.
        """)

# ──────────────────────────────────────────────────────────────────────────────
# TAB 3: HOW IT WORKS
# ──────────────────────────────────────────────────────────────────────────────

with tab3:
    st.subheader("🏗️ Model Architecture")

    st.markdown("""
    ### Two-Stage Training Pipeline

    #### Stage 1: Masked Autoencoder (MAE) Pre-training
    - **Task**: Self-supervised learning on unlabeled dermoscopy images
    - **Method**: Masks 75% of image patches, learns to reconstruct them
    - **Goal**: Learn rich visual representations of skin lesions
    - **Reference**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (CVPR 2022)

    #### Stage 2: Supervised Fine-tuning
    - **Base**: Pre-trained ViT encoder (from MAE)
    - **Loss**: Focal loss (γ=2) to handle class imbalance
    - **Optimization**: Layer-wise LR decay (LLRD=0.75)
    - **Data**: Weighted random sampling (oversample rare classes)
    - **Regularization**: Modality dropout (p=0.1) for robustness

    ### Features
    - **Multimodal fusion**: Image + clinical data (age, sex, localization, etc.)
    - **Late fusion**: Separate encoders → concatenate → MLP head
    - **Rare disease focus**: Explicit focal loss on rare classes
    """)

    st.divider()

    st.subheader("🚀 Quick Start")
    st.code("""
# 1. Install dependencies
poetry install

# 2. Download datasets (see scripts/download_data.py for instructions)
make download

# 3. Train Stage 1 (MAE pre-training)
make train-s1

# 4. Train Stage 2 (fine-tuning)
make train-s2

# 5. Run API and frontend
make api      # terminal 1
make frontend # terminal 2
    """, language="bash")

    st.divider()

    st.subheader("📚 Resources")
    st.markdown("""
    - **GitHub**: [RareSight](https://github.com/yourusername/raresight)
    - **ISIC**: [isic-archive.com](https://isic-archive.com)
    - **MAE Paper**: [arxiv.org/abs/2111.06377](https://arxiv.org/abs/2111.06377)
    - **ViT**: [arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
    """)
