from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from llm.gemini import generate_feedback
from model.inference import InventoryClassifier
from prompts.presets import PRESET_EXPECTED_ITEMS


load_dotenv()

st.set_page_config(page_title="Inventory Checker", page_icon=":package:", layout="centered")
st.title("Inventory Checker")
st.caption("Upload an image to detect items, then get contextual feedback.")

ARTIFACT_PATH = Path("best_model_with_thresholds.pt")

if not ARTIFACT_PATH.exists():
    st.error(f"Model artifact not found: {ARTIFACT_PATH.resolve()}")
    st.stop()


@st.cache_resource
def load_classifier() -> InventoryClassifier:
    return InventoryClassifier(ARTIFACT_PATH)


classifier: InventoryClassifier = load_classifier()

preset = st.selectbox("Choose a preset", options=list(PRESET_EXPECTED_ITEMS.keys()))
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", width="stretch")

    with st.spinner("Running model inference..."):
        result = classifier.predict_from_pil(image)

    st.subheader("Detected Labels")
    if result.detected_labels:
        st.write(", ".join(result.detected_labels))
    else:
        st.write("No labels detected above class thresholds.")

    with st.expander("Model probabilities and thresholds"):
        for label in classifier.class_names:
            st.write(
                f"{label}: prob={result.probabilities[label]:.3f}, "
                f"threshold={result.thresholds[label]:.2f}"
            )

    labels_key = tuple(result.detected_labels)
    feedback_key = (preset, labels_key)
    if "feedback_key" not in st.session_state:
        st.session_state.feedback_key = None
    if "feedback_text" not in st.session_state:
        st.session_state.feedback_text = ""

    if st.session_state.feedback_key != feedback_key:
        with st.spinner("Generating contextual feedback..."):
            st.session_state.feedback_text = generate_feedback(result.detected_labels, preset)
        st.session_state.feedback_key = feedback_key

    if st.button("Refresh AI Feedback"):
        with st.spinner("Refreshing contextual feedback..."):
            st.session_state.feedback_text = generate_feedback(result.detected_labels, preset)

    st.subheader("Contextual Feedback")
    st.write(st.session_state.feedback_text)
