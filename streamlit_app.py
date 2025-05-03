import streamlit as st
import os
import json
import joblib

from testing import evaluate_utterance  # Per-utterance evaluation
from main import main as run_full_evaluation  # Full dataset evaluation

# ---- Paths ----
MFCC_PATH = "mfcc_with_phoneme_indices.json"
PHONEME_MAP_PATH = "phoneme_to_index.json"
MODELS_DIR = "hmm_models"


# ---- Load phoneme models ----
def load_phoneme_models(models_dir):
    phoneme_models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl"):
            phoneme_idx = int(filename.split(".")[0])
            model = joblib.load(os.path.join(models_dir, filename))
            phoneme_models[phoneme_idx] = model
    return phoneme_models


# ---- Load JSON data ----
def load_data(mfcc_path, phoneme_map_path):
    with open(mfcc_path, "r") as f:
        mfcc_data = json.load(f)
    with open(phoneme_map_path, "r") as f:
        phoneme_to_index = json.load(f)
    return mfcc_data, phoneme_to_index


# ---- Streamlit Interface ----
st.title("🔊 HMM-based ASR Phoneme Evaluation")
st.markdown("Input an utterance index to evaluate the predicted vs actual phoneme sequence.")

# ---- Load Data and Models ----
with st.spinner("Loading models and data..."):
    phoneme_models = load_phoneme_models(MODELS_DIR)
    mfcc_data, phoneme_to_index = load_data(MFCC_PATH, PHONEME_MAP_PATH)

# ---- Per-Utterance Evaluation ----
index = st.number_input("Utterance Index", min_value=0, max_value=len(mfcc_data) - 1, step=1)

if st.button("Evaluate Utterance"):
    actual, predicted, accuracy, error = evaluate_utterance(
        index,
        MFCC_PATH,
        PHONEME_MAP_PATH,
        phoneme_models
    )

    if error:
        st.error("❌ Error during evaluation:")
        st.code(error)
    elif actual is None:
        st.warning(predicted)  # `predicted` holds an error message if actual is None
    else:
        st.success("✅ Evaluation completed!")
        st.markdown(f"**Accuracy:** `{accuracy:.2f}%`")
        st.markdown("**Actual Phonemes:**")
        st.code(" ".join(actual))
        st.markdown("**Predicted Phonemes:**")
        st.code(" ".join(predicted))

# ---- Full Dataset Evaluation ----
st.markdown("---")
st.subheader("📊 Evaluate Entire Dataset")

if st.button("Run Full 1-Hour Evaluation"):
    with st.spinner("Running full evaluation..."):
        try:
            full_accuracy = run_full_evaluation()
            st.success("🎉 Full evaluation completed!")
            st.markdown(f"**Overall Phoneme Match Accuracy:** `{full_accuracy:.2f}%`")
        except Exception as e:
            st.error("❌ Error during full evaluation:")
            st.code(str(e))
