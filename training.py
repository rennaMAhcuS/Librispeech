"""
Improved HMM-based ASR System for Phoneme Recognition - Training Module
This module provides functions to train HMM models for phoneme recognition.
"""

import os
import json
import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
import joblib


# ================ TRAINING CODE ================

def train_phoneme_hmms(mfcc_with_phonemes_path, phoneme_to_index_path, models_dir="hmm_models"):
    """
    Train HMM models for each phoneme using MFCC features
    """
    # Load phoneme mapping
    with open(phoneme_to_index_path, "r") as f:
        phoneme_to_index = json.load(f)

    # Convert string keys to integers for the reverse mapping
    index_to_phoneme = {int(v): k for k, v in phoneme_to_index.items()}

    # Load phoneme-annotated MFCC data
    with open(mfcc_with_phonemes_path, "r") as f:
        mfcc_with_phonemes = json.load(f)

    # Group MFCCs per phoneme
    phoneme_data = {int(idx): [] for idx in phoneme_to_index.values()}

    for item in tqdm(mfcc_with_phonemes, desc="Organizing training data"):
        mfcc = np.array(item["mfcc"])
        phoneme_indices = item["phoneme_indices"]

        # Skip samples with no phonemes or insufficient MFCC frames
        if len(phoneme_indices) == 0 or mfcc.shape[0] < len(phoneme_indices):
            continue

        n_frames = mfcc.shape[0]
        chunk_size = n_frames // len(phoneme_indices)

        for i, idx in enumerate(phoneme_indices):
            idx = int(idx)  # Ensure integer index
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < len(phoneme_indices) - 1 else n_frames
            segment = mfcc[start:end]

            # Only add valid segments with the correct feature dimension
            if segment.shape[0] > 0 and segment.shape[1] == 13:
                phoneme_data[idx].append(segment)

    # Train a 3-state HMM for each phoneme
    phoneme_models = {}
    os.makedirs(models_dir, exist_ok=True)

    for phoneme_idx, sequences in tqdm(phoneme_data.items(), desc="Training HMMs"):
        # Skip phonemes with insufficient training data
        if len(sequences) < 5:
            print(f"Skipping phoneme {phoneme_idx} ({index_to_phoneme.get(phoneme_idx, '?')}) - insufficient data")
            continue

        try:
            # Stack sequences for training
            x = np.vstack(sequences)
            lengths = [len(seq) for seq in sequences]

            model = hmm.GaussianHMM(
                n_components=3,
                covariance_type='diag',
                n_iter=50,
                verbose=False
            )
            model.fit(x, lengths)

            # Save the trained model
            phoneme_models[phoneme_idx] = model
            joblib.dump(model, f"{models_dir}/{phoneme_idx}.pkl")
            print(f"✅ Trained model for phoneme {phoneme_idx} ({index_to_phoneme.get(phoneme_idx, '?')})")

        except Exception as e:
            print(f"❌ Failed for phoneme {phoneme_idx} ({index_to_phoneme.get(phoneme_idx, '?')}): {e}")

    print(f"✅ Trained {len(phoneme_models)} phoneme models")
    return phoneme_models, index_to_phoneme


def main():
    # Example usage
    mfcc_with_phonemes_path = "mfcc_with_phoneme_indices.json"
    phoneme_to_index_path = "phoneme_to_index.json"

    # Train models with improved parameters
    return train_phoneme_hmms(mfcc_with_phonemes_path, phoneme_to_index_path)


if __name__ == "__main__":
    main()
