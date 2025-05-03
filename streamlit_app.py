import streamlit as st
import os
import json
import joblib
from testing import evaluate_utterance
from main import main as run_full_evaluation

# ---- Paths ----
MFCC_PATH = "mfcc_with_phoneme_indices.json"  # Updated path name
PHONEME_MAP_PATH = "phoneme_to_index.json"
INVERSE_DICT_PATH = "inverse_phoneme_dict.json"
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

# ---- Load JSON files ----
def load_data(mfcc_path, phoneme_map_path, inverse_dict_path):
    with open(mfcc_path, "r") as f:
        mfcc_data = json.load(f)
    with open(phoneme_map_path, "r") as f:
        phoneme_to_index = json.load(f)
    with open(inverse_dict_path, "r") as f:
        inverse_phoneme_dict = json.load(f)
    index_to_phoneme = {v: k for k, v in phoneme_to_index.items()}
    return mfcc_data, phoneme_to_index, index_to_phoneme, inverse_phoneme_dict

# ---- Phoneme to Word Decoder ----
def phonemes_to_words(predicted_phonemes, inverse_dict, max_n=5):
    i = 0
    result = []
    while i < len(predicted_phonemes):
        matched = False
        for n in range(min(max_n, len(predicted_phonemes) - i), 0, -1):
            ngram = " ".join(predicted_phonemes[i:i + n])
            if ngram in inverse_dict:
                # Get the word from inverse_dict and ensure it's a string
                word = inverse_dict[ngram]
                if isinstance(word, list):  # If it's a list, take the first item
                    if word:  # Check if a list is not empty
                        word = word[0]  # Take the first option
                    else:
                        word = ngram  # Fallback to the original ngram
                result.append(str(word))  # Convert to string just to be safe
                i += n
                matched = True
                break
        if not matched:
            # Skip this phoneme
            i += 1
    return result

# ---- Streamlit UI ----
st.title("🔊 HMM-based ASR Phoneme Evaluation")
st.markdown("Evaluate predicted vs actual phonemes and decode them to words.")

with st.spinner("Loading models and data..."):
    phoneme_models = load_phoneme_models(MODELS_DIR)
    mfcc_data, phoneme_to_index, index_to_phoneme, inverse_dict = load_data(
        MFCC_PATH, PHONEME_MAP_PATH, INVERSE_DICT_PATH
    )

# ---- Utterance Selection ----
index = st.number_input(f"Utterance Index (Range: 0 to {len(mfcc_data) - 1})", min_value=0, max_value=len(mfcc_data) - 1, step=1)
if st.button("🎯 Evaluate Utterance"):
    actual, predicted, accuracy, error = evaluate_utterance(
        index, MFCC_PATH, PHONEME_MAP_PATH, phoneme_models
    )

    if error:
        st.error("❌ Error during evaluation:")
        st.code(error)
    elif actual is None:
        st.warning(predicted)
    else:
        st.success("✅ Evaluation completed!")
        st.markdown(f"**Accuracy:** `{accuracy:.2f}%`")
        st.markdown("**Actual Phonemes:**")
        st.code(" ".join(actual))
        st.markdown("**Predicted Phonemes:**")
        st.code(" ".join(predicted))

        # --- Audio Playback using a stored path ---
        audio_path = mfcc_data[index]["path"]
        if os.path.exists(audio_path):
            st.audio(audio_path, format="audio/wav")
        else:
            st.warning(f"⚠️ Audio file not found: {audio_path}")

        # --- Word Decoding ---
        try:
            decoded_words = phonemes_to_words(predicted, inverse_dict)
            if decoded_words:
                # Handle any potential type issues with join
                try:
                    decoded_sentence = " ".join(decoded_words)
                    st.markdown("**Predicted Sentence:**")
                    st.markdown(f"> {decoded_sentence.capitalize()}.")
                except TypeError as e:
                    # Fallback: manually convert each item to string
                    decoded_sentence = " ".join([str(word) for word in decoded_words])
                    st.markdown("**Predicted Sentence:**")
                    st.markdown(f"> {decoded_sentence.capitalize()}.")
            else:
                st.info("Could not decode phonemes into words.")
        except Exception as e:
            st.error(f"Error in word decoding: {str(e)}")

# ---- Full Evaluation ----
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
