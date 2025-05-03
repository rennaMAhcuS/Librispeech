import streamlit as st
import os
import json
import numpy as np
import joblib
from main import evaluate_utterance  # <-- Import your function from your own file

# ---- Paths ----
MFCC_PATH = "mfcc_with_phoneme_indices.json"
PHONEME_MAP_PATH = "phoneme_to_index.json"
MODELS_DIR = "hmm_models"

# ---- Load phoneme models ----
@st.cache_resource
def load_phoneme_models(models_dir):
    phoneme_models = {}
    for filename in os.listdir(models_dir):
        if filename.endswith(".pkl"):
            phoneme_idx = int(filename.split(".")[0])
            model = joblib.load(os.path.join(models_dir, filename))
            phoneme_models[phoneme_idx] = model
    return phoneme_models

# ---- Load JSON data ----
@st.cache_data
def load_data(mfcc_path, phoneme_map_path):
    with open(mfcc_path, "r") as f:
        mfcc_data = json.load(f)
    with open(phoneme_map_path, "r") as f:
        phoneme_to_index = json.load(f)
    return mfcc_data, phoneme_to_index

# ---- Streamlit Interface ----
st.title("🔊 HMM-based ASR Phoneme Evaluation")
st.markdown("Input an utterance index to evaluate the predicted vs actual phoneme sequence.")

with st.spinner("Loading models and data..."):
    phoneme_models = load_phoneme_models(MODELS_DIR)
    mfcc_data, phoneme_to_index = load_data(MFCC_PATH, PHONEME_MAP_PATH)

# Input
index = st.number_input("Utterance Index", min_value=0, max_value=len(mfcc_data) - 1, step=1)

if st.button("Evaluate"):
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
        st.warning(predicted)  # `predicted` holds error message if actual is None
    else:
        st.success("✅ Evaluation completed!")
        st.markdown(f"**Accuracy:** `{accuracy:.2f}%`")
        st.markdown("**Actual Phonemes:**")
        st.code(" ".join(actual))
        st.markdown("**Predicted Phonemes:**")
        st.code(" ".join(predicted))
