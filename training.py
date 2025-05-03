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
from sklearn.cluster import KMeans

# ================ TRAINING CODE ================

def normalize_mfcc(mfcc_features):
    """Normalize MFCC features for better model convergence"""
    mean = np.mean(mfcc_features, axis=0)
    std = np.std(mfcc_features, axis=0)
    return (mfcc_features - mean) / (std + 1e-8)  # Add small epsilon to avoid division by zero

def estimate_phoneme_durations(mfcc_with_phonemes, index_to_phoneme):
    """Estimate average duration of each phoneme from the data"""
    phoneme_durations = {}
    phoneme_counts = {}
    
    for item in mfcc_with_phonemes:
        mfcc = np.array(item["mfcc"])
        phoneme_indices = item["phoneme_indices"]
        
        if len(phoneme_indices) == 0 or mfcc.shape[0] < len(phoneme_indices):
            continue
            
        n_frames = mfcc.shape[0]
        avg_duration = n_frames / len(phoneme_indices)
        
        for idx in phoneme_indices:
            idx = int(idx)
            phoneme = index_to_phoneme.get(str(idx), "UNK")
            
            if phoneme not in phoneme_durations:
                phoneme_durations[phoneme] = 0
                phoneme_counts[phoneme] = 0
                
            phoneme_durations[phoneme] += avg_duration
            phoneme_counts[phoneme] += 1
    
    # Calculate average durations
    avg_durations = {}
    default_duration = 5  # Default duration for unseen phonemes
    
    for phoneme, total_duration in phoneme_durations.items():
        if phoneme_counts[phoneme] > 0:
            avg_durations[phoneme] = total_duration / phoneme_counts[phoneme]
        else:
            avg_durations[phoneme] = default_duration
            
    return avg_durations

def assign_variable_length_segments(mfcc, phoneme_indices, index_to_phoneme, avg_durations):
    """Assign variable-length segments to phonemes based on their average durations"""
    n_frames = mfcc.shape[0]
    
    # Get expected duration for each phoneme
    durations = [avg_durations.get(index_to_phoneme.get(str(idx), "UNK"), 5) for idx in phoneme_indices]
    
    # Scale to match total frames
    total_expected = sum(durations)
    scale_factor = n_frames / total_expected
    
    segments = []
    current_pos = 0
    
    for i, idx in enumerate(phoneme_indices):
        # Calculate this phoneme's segment duration
        duration = int(durations[i] * scale_factor)
        
        # Handle last phoneme specially to avoid rounding errors
        if i == len(phoneme_indices) - 1:
            end_pos = n_frames
        else:
            end_pos = min(current_pos + duration, n_frames)
            
        # Add segment if it has valid length
        if end_pos > current_pos:
            segments.append((idx, mfcc[current_pos:end_pos]))
            current_pos = end_pos
    
    return segments

def initialize_hmm(n_states=3, n_features=13):
    """Initialize an HMM with a left-to-right topology"""
    # Initialize transition matrix (left-to-right)
    transmat = np.zeros((n_states, n_states))
    for i in range(n_states-1):
        transmat[i, i] = 0.6    # Self-transition probability
        transmat[i, i+1] = 0.4  # Forward transition probability
    transmat[n_states-1, n_states-1] = 1.0  # Last state always self-loops
    
    # Initialize start probability (always start in first state)
    startprob = np.zeros(n_states)
    startprob[0] = 1.0
    
    # Initialize means randomly
    means = np.random.randn(n_states, n_features) * 0.2
    
    # Initialize covariances (diagonal)
    covars = np.ones((n_states, n_features)) * 1.0
    
    return startprob, transmat, means, covars

def train_phoneme_hmms(mfcc_with_phonemes_path, phoneme_to_index_path, models_dir="hmm_models"):
    """
    Train HMM models for each phoneme using MFCC features
    """
    # Load phoneme mapping
    with open(phoneme_to_index_path, "r") as f:
        phoneme_to_index = json.load(f)
    
    # Convert string keys to integers for the reverse mapping
    index_to_phoneme = {str(v): k for k, v in phoneme_to_index.items()}
    
    # Load phoneme-annotated MFCC data
    with open(mfcc_with_phonemes_path, "r") as f:
        mfcc_with_phonemes = json.load(f)
    
    # Estimate average phoneme durations
    print("Estimating phoneme durations...")
    avg_durations = estimate_phoneme_durations(mfcc_with_phonemes, index_to_phoneme)
    
    # Group MFCCs per phoneme using improved segmentation
    print("Organizing training data with improved segmentation...")
    phoneme_data = {}
    
    for item in tqdm(mfcc_with_phonemes, desc="Segmenting data"):
        mfcc = np.array(item["mfcc"])
        phoneme_indices = item["phoneme_indices"]
        
        # Skip samples with no phonemes or insufficient MFCC frames
        if len(phoneme_indices) == 0 or mfcc.shape[0] < len(phoneme_indices):
            continue
        
        # Normalize MFCC features
        normalized_mfcc = normalize_mfcc(mfcc)
        
        # Assign variable-length segments to phonemes
        segments = assign_variable_length_segments(
            normalized_mfcc, phoneme_indices, index_to_phoneme, avg_durations
        )
        
        # Group by phoneme index
        for idx, segment in segments:
            idx = int(idx)
            if idx not in phoneme_data:
                phoneme_data[idx] = []
                
            # Only add segments with sufficient frames and correct feature dimension
            if segment.shape[0] >= 3 and segment.shape[1] == 13:
                phoneme_data[idx].append(segment)
    
    # Train a 3-state HMM for each phoneme
    phoneme_models = {}
    os.makedirs(models_dir, exist_ok=True)
    
    for phoneme_idx, sequences in tqdm(phoneme_data.items(), desc="Training HMMs"):
        # Skip phonemes with insufficient training data
        if len(sequences) < 5:
            print(f"Skipping phoneme {phoneme_idx} ({index_to_phoneme.get(str(phoneme_idx), '?')}) - insufficient data")
            continue
        
        try:
            # Stack sequences for training
            X = np.vstack(sequences)
            lengths = [len(seq) for seq in sequences]
            
            # Initialize HMM parameters with left-to-right topology
            n_states = 3
            n_features = 13
            startprob, transmat, means, covars = initialize_hmm(n_states, n_features)
            
            # Initialize model with our custom parameters
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=100,
                init_params="",  # Don't initialize any params - we'll set them manually
                verbose=True
            )
            
            # Set our custom parameters
            model.startprob_ = startprob
            model.transmat_ = transmat
            
            # Use K-means to better initialize the means
            if len(X) > n_states * 5:  # Only use K-means if we have enough data
                kmeans = KMeans(n_clusters=n_states, random_state=42)
                kmeans.fit(X)
                means = kmeans.cluster_centers_
            
            model.means_ = means
            model.covars_ = covars
            
            # Fit model with our constraints
            model.fit(X, lengths)
            
            # Save the trained model
            phoneme_models[phoneme_idx] = model
            joblib.dump(model, f"{models_dir}/{phoneme_idx}.pkl")
            print(f"✅ Trained model for phoneme {phoneme_idx} ({index_to_phoneme.get(str(phoneme_idx), '?')})")
            
        except Exception as e:
            print(f"❌ Failed for phoneme {phoneme_idx} ({index_to_phoneme.get(str(phoneme_idx), '?')}): {e}")
    
    # Save the phoneme durations for use during decoding
    with open(f"{models_dir}/phoneme_durations.json", "w") as f:
        json.dump(avg_durations, f, indent=2)
    
    print(f"✅ Trained {len(phoneme_models)} phoneme models")
    return phoneme_models, index_to_phoneme

def train_triphone_models(mfcc_with_phonemes_path, phoneme_to_index_path, models_dir="triphone_models"):
    """
    Train context-dependent triphone models for improved accuracy
    A triphone is a phoneme with left and right context
    """
    print("Training context-dependent triphone models...")
    # Load phoneme mapping
    with open(phoneme_to_index_path, "r") as f:
        phoneme_to_index = json.load(f)
    
    # Convert string keys to integers for the reverse mapping
    index_to_phoneme = {str(v): k for k, v in phoneme_to_index.items()}
    
    # Load phoneme-annotated MFCC data
    with open(mfcc_with_phonemes_path, "r") as f:
        mfcc_with_phonemes = json.load(f)
    
    # Organize triphone data
    triphone_data = {}  # Format: {(left_idx, center_idx, right_idx): [segments]}
    
    for item in tqdm(mfcc_with_phonemes, desc="Organizing triphone data"):
        mfcc = np.array(item["mfcc"])
        phoneme_indices = item["phoneme_indices"]
        
        # Skip samples with too few phonemes
        if len(phoneme_indices) < 3:
            continue
        
        # Normalize MFCC features
        normalized_mfcc = normalize_mfcc(mfcc)
        
        # Create triphones with context
        for i in range(1, len(phoneme_indices)-1):
            left_idx = int(phoneme_indices[i-1])
            center_idx = int(phoneme_indices[i])
            right_idx = int(phoneme_indices[i+1])
            
            triphone_key = (left_idx, center_idx, right_idx)
            
            # Simple segmentation for triphone (can be improved)
            n_frames = mfcc.shape[0]
            frame_per_phoneme = n_frames // len(phoneme_indices)
            start = i * frame_per_phoneme
            end = (i + 1) * frame_per_phoneme
            
            segment = normalized_mfcc[start:end]
            
            if segment.shape[0] >= 3 and segment.shape[1] == 13:
                if triphone_key not in triphone_data:
                    triphone_data[triphone_key] = []
                triphone_data[triphone_key].append(segment)
    
    # Train models for frequent triphones
    os.makedirs(models_dir, exist_ok=True)
    triphone_models = {}
    triphone_to_id = {}
    id_to_triphone = {}
    next_id = 0
    
    print(f"Found {len(triphone_data)} unique triphones")
    
    for triphone_key, sequences in tqdm(triphone_data.items(), desc="Training triphone HMMs"):
        # Only train models for triphones with sufficient data
        if len(sequences) < 10:
            continue
            
        # Assign an ID to this triphone
        triphone_to_id[triphone_key] = next_id
        id_to_triphone[next_id] = triphone_key
        triphone_id = next_id
        next_id += 1
        
        # Format a readable name for this triphone
        left, center, right = triphone_key
        left_ph = index_to_phoneme.get(str(left), "?")
        center_ph = index_to_phoneme.get(str(center), "?")
        right_ph = index_to_phoneme.get(str(right), "?")
        triphone_name = f"{left_ph}-{center_ph}+{right_ph}"
        
        try:
            # Stack sequences for training
            X = np.vstack(sequences)
            lengths = [len(seq) for seq in sequences]
            
            # Initialize HMM parameters
            n_states = 3
            n_features = 13
            startprob, transmat, means, covars = initialize_hmm(n_states, n_features)
            
            # Initialize model with our custom parameters
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=50,
                init_params="",
                verbose=False
            )
            
            # Set our custom parameters
            model.startprob_ = startprob
            model.transmat_ = transmat
            model.means_ = means
            model.covars_ = covars
            
            # Fit model
            model.fit(X, lengths)
            
            # Save the trained model
            triphone_models[triphone_id] = model
            joblib.dump(model, f"{models_dir}/{triphone_id}.pkl")
            print(f"✅ Trained model for triphone {triphone_id} ({triphone_name})")
            
        except Exception as e:
            print(f"❌ Failed for triphone {triphone_name}: {e}")
    
    # Save mapping between triphone IDs and phoneme contexts
    with open(f"{models_dir}/triphone_mapping.json", "w") as f:
        # Convert tuple keys to strings for JSON
        mapping = {str(k): v for k, v in triphone_to_id.items()}
        json.dump(mapping, f, indent=2)
    
    print(f"✅ Trained {len(triphone_models)} triphone models")
    return triphone_models, id_to_triphone

def save_model_metadata(models_dir, index_to_phoneme):
    """Save metadata for models to assist in decoding"""
    metadata = {
        "phonemes": index_to_phoneme,
        "model_type": "hmm",
        "feature_type": "mfcc",
        "n_states": 3,
        "n_features": 13,
        "version": "1.0"
    }
    
    with open(f"{models_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def main():
    # Paths to data files
    mfcc_with_phonemes_path = "mfcc_with_phoneme_indices.json"
    phoneme_to_index_path = "phoneme_to_index.json"
    
    # Train phoneme models with improved parameters
    phoneme_models, index_to_phoneme = train_phoneme_hmms(
        mfcc_with_phonemes_path, 
        phoneme_to_index_path,
        models_dir="hmm_models"
    )
    
    # Save model metadata
    save_model_metadata("hmm_models", index_to_phoneme)
    
    # Optionally train triphone models for better accuracy
    train_triphones = False  # Set to True to enable triphone training
    if train_triphones:
        triphone_models, triphone_mapping = train_triphone_models(
            mfcc_with_phonemes_path,
            phoneme_to_index_path,
            models_dir="triphone_models"
        )
        
    print("Training complete!")

if __name__ == "__main__":
    main()