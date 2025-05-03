# Cell: Import Libraries
import os
import requests
import tarfile
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
from glob import glob
import python_speech_features as psf
from scipy.io.wavfile import read
from IPython.display import Audio
import json
import g2p_en
import nltk
from collections import defaultdict
import pickle
import shutil
from hmmlearn import hmm
from itertools import groupby

# Specify the download directory
download_dir = "C:/Users/abhin/nltk_data"

# Download both resources to the specified directory
nltk.download('punkt', download_dir=download_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=download_dir)

# URL for the dataset
url = "http://www.openslr.org/resources/12/train-clean-100.tar.gz" 

# Destination path for downloading
download_path = "train-clean-100.tar.gz"

# Function to download a file with progress bar
def download_file(url, destination_path):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(destination_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024), total=total_size // 1024, unit='KB', desc="Downloading"):
                if chunk:
                    f.write(chunk)

# # Download the file
# download_file(url, download_path)

# Function to extract tar.gz file
def extract_file(tar_path, dest_dir):
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=dest_dir)
        print(f"Dataset extracted to {dest_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")

# Download the file
if not os.path.exists(download_path):
    download_file(url, download_path)

# Define the destination extraction folder in the current working directory
extracted_dir = os.path.join(os.getcwd(), "audio_files")

# Extract the tar.gz file if not already extracted
if os.path.exists(download_path) and not os.path.exists(extracted_dir):
    extract_file(download_path, extracted_dir)
else:
    if os.path.exists(extracted_dir):
        print(f"Dataset already extracted at {extracted_dir}")
    else:
        print(f"Download path does not exist: {download_path}")
        

# Cell: Locate all .flac files
flac_files = glob(os.path.join(extracted_dir, "**", "*.flac"), recursive=True)
print(f"Found {len(flac_files)} .flac files")
print("Example file:", flac_files[0])

# Path settings
source_root = "audio_files/LibriSpeech/train-clean-100"
wav_root = "train-clean-100-wav"

# Find all .flac files under the source directory
flac_files = []
for root, _, files in os.walk(source_root):
    for file in files:
        if file.endswith(".flac"):
            flac_files.append(os.path.join(root, file))

# Convert and save .wav files under simplified structure
def convert_flac_to_wav(flac_files, source_root, wav_root):
    for flac_path in tqdm(flac_files, desc="Converting to WAV"):
        # Get relative path from source_root (e.g., 19/198/19-198-0000.flac)
        relative_path = os.path.relpath(flac_path, source_root)
        wav_path = os.path.join(wav_root, os.path.splitext(relative_path)[0] + ".wav")
        
        # Create directory if needed
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        
        # Load and write audio
        audio, sr = sf.read(flac_path)
        sf.write(wav_path, audio, sr)
        print(f"Saved: {wav_path}")

# convert_flac_to_wav(flac_files, source_root, wav_root)

# Function to copy transcripts to the new structure
def copy_transcripts(source_root, wav_root):
    for root, _, files in tqdm(os.walk(source_root), desc="Copying Transcripts"):
        for file in files:
            if file.endswith(".trans.txt"):
                # Full path to the transcript file
                transcript_file = os.path.join(root, file)
                
                # Get the relative path from source_root (e.g., 19/198/19-198.trans.txt)
                relative_path = os.path.relpath(root, source_root)
                
                # Create the corresponding directory in the new location
                dest_dir = os.path.join(wav_root, relative_path)
                os.makedirs(dest_dir, exist_ok=True)
                
                # Construct the destination path for the transcript file
                transcript_dest = os.path.join(dest_dir, file)
                
                # Copy the transcript file
                shutil.copy(transcript_file, transcript_dest)
                print(f"Copied: {transcript_dest}")

# Call the function to copy transcripts
# copy_transcripts(source_root, wav_root)

import glob
wav_files = glob.glob("train-clean-100-wav/**/*.wav", recursive=True)
print(f"Found {len(wav_files)} .wav files")

# Cell: Extract MFCC features

def extract_mfcc(wav_path, num_mfcc=13):
    sr, signal = read(wav_path)  # Correct order: sr, signal from scipy read
    mfcc_feat = psf.mfcc(signal, sr, numcep=num_mfcc)
    return mfcc_feat

# Example on one file
sample_mfcc = extract_mfcc(wav_files[0])
print("MFCC shape:", sample_mfcc.shape)

# Cell: Visualize waveform and MFCCs
def visualize_waveform_and_mfcc(wav_path):
    signal, sr = librosa.load(wav_path, sr=None)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title("Waveform")
    
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfcc.T, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCC")
    
    plt.tight_layout()
    plt.show()

# Test on one file
visualize_waveform_and_mfcc(wav_files[0])

import random
# wav_files = [...]

# Limit to ~1 hour of data (assuming ~6 minutes per file on average)
target_duration = 60 * 60  # 1 hour in seconds
selected_files = []

# Randomly shuffle the files before selection
random.shuffle(wav_files)

current_duration = 0
i = 0

# Randomly select files until we reach 1 hour
while current_duration < target_duration:
    file = wav_files[i]  # Select a file
    signal, sr = librosa.load(file, sr=None)  # Load the file
    current_duration += len(signal) / sr
    selected_files.append(file)
    i += 1

# Save the selected files list
with open("selected_files.json", "w") as f:
    json.dump(selected_files, f)

print(f"Selected {len(selected_files)} files, total duration: {current_duration / 60:.2f} minutes")

# Load the selected files
with open("selected_files.json", "r") as f:
    selected_files = json.load(f)

# Extract MFCCs
mfcc_data = []  # list of dicts: {"path": ..., "mfcc": ...}

for wav_path in tqdm(selected_files, desc="Extracting MFCCs"):
    try:
        # Load the signal using soundfile (sf)
        signal, sr = sf.read(wav_path)
        
        # Check if signal is a numpy array (1D for mono audio)
        if not isinstance(signal, np.ndarray) or len(signal.shape) != 1:
            print(f"Warning: {wav_path} has an invalid signal format")
            continue
        
        # Compute MFCC features using python_speech_features
        mfcc_feat = psf.mfcc(signal, sr, numcep=13)

        # Check if mfcc_feat is a 2D numpy array (valid MFCC features)
        if not isinstance(mfcc_feat, np.ndarray) or len(mfcc_feat.shape) != 2:
            print(f"Warning: {wav_path} returned invalid MFCC features")
            continue
        
        # Append the MFCC data to the list
        mfcc_data.append({
            "path": wav_path,
            "mfcc": mfcc_feat.tolist()
        })
        
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")

# Save MFCC data to a file
with open("mfcc_features_subset.json", "w") as f:
    json.dump(mfcc_data, f)

print("Saved MFCC features to mfcc_features_subset.json")

from g2p_en import G2p

g2p = G2p()

# Example word to phoneme conversion
word = "hello"
phonemes = g2p(word)
print(f"Word: {word} -> Phonemes: {phonemes}")

import re

def clean_transcript(transcript):
    # Lowercase and remove unwanted characters (punctuation, special symbols)
    transcript = transcript.lower()  # Make lowercase
    transcript = re.sub(r'[^a-z\s]', '', transcript)  # Remove non-alphabetical characters
    return transcript

# Example
transcript = "Hello, world! How are you?"
cleaned_transcript = clean_transcript(transcript)
print(f"Cleaned Transcript: {cleaned_transcript}")

import os

# Function to get the transcript for a given wav file
def get_transcript_for_wav(wav_path):
    # Extract the speaker and chapter info from the path
    speaker_id = wav_path.split("\\")[1]  # Example: '19'
    chapter_id = wav_path.split("\\")[2]  # Example: '198'
    audio_id = wav_path.split("\\")[-1].replace(".wav", "")  # Example: '19-198-0001'

    # Construct the transcript filename based on speaker_id and chapter_id
    transcript_file = os.path.join("train-clean-100-wav", speaker_id, chapter_id, f"{speaker_id}-{chapter_id}.trans.txt")

    # Read the transcript file and find the corresponding line for the audio file
    try:
        with open(transcript_file, "r") as f:
            lines = f.readlines()
        
        # Find the line corresponding to the audio_id (audio file name)
        for line in lines:
            if audio_id in line:
                # Split the line to extract the actual transcript (skip the first column)
                transcript = line.split(" ", 1)[1].strip()
                return transcript
    except FileNotFoundError:
        print(f"Transcript file {transcript_file} not found.")
        return None  # If the transcript file is not found, return None

    return None  # Return None if no matching transcript is found

# Initialize G2p model
g2p = G2p()

# Step 1: Clean the transcript by removing unnecessary characters and normalizing the text
def clean_transcript(transcript):
    # Convert to lowercase and remove punctuation (except spaces)
    transcript = transcript.lower()
    transcript = re.sub(r"[^a-zA-Z\s]", "", transcript)
    return transcript

# Step 2: Convert a cleaned transcript to phonemes with stress stripped
def convert_to_phonemes(cleaned_transcript):
    words = cleaned_transcript.split()
    phonemes = []

    for word in words:
        # Get phoneme representation for each word
        raw_phonemes = g2p(word)

        # Strip stress markers (e.g., AA1 -> AA)
        normalized = [re.sub(r'\d$', '', p) for p in raw_phonemes]
        phonemes.extend(normalized)

    return phonemes

# Load the MFCC data
with open("mfcc_features_subset.json", "r") as f:
    mfcc_data = json.load(f)

# Create a list to hold MFCC data with phoneme mappings
mfcc_with_phonemes = []

# Map MFCC features to phonemes
for data in tqdm(mfcc_data, desc="Mapping MFCCs to Phonemes"):
    wav_path = data["path"]
    mfcc_feat = data["mfcc"]
    
    # Retrieve the corresponding transcript for the audio file
    transcript = get_transcript_for_wav(wav_path)
    if transcript is None:
        continue  # Skip if no transcript is found
    
    # Clean the transcript
    cleaned_transcript = clean_transcript(transcript)
    
    # Convert the cleaned transcript to phonemes
    phonemes = convert_to_phonemes(cleaned_transcript)
    
    # Append the MFCC features and the phoneme sequence
    mfcc_with_phonemes.append({
        "path": wav_path,
        "mfcc": mfcc_feat,
        "phonemes": phonemes
    })

# Save the new data (MFCC + phonemes)
with open("mfcc_with_phonemes.json", "w") as f:
    json.dump(mfcc_with_phonemes, f)

print("Mapped MFCCs to Phonemes and saved.")

# Create a phoneme to index mapping (dictionary)
phoneme_set = set()  # To store unique phonemes

# Collect all unique phonemes from the mapped data
for data in mfcc_with_phonemes:
    phonemes = data["phonemes"]
    phoneme_set.update(phonemes)

# Create the phoneme-to-index mapping
phoneme_to_index = {phoneme: idx for idx, phoneme in enumerate(sorted(phoneme_set))}

# Save the phoneme-to-index mapping for later use
with open("phoneme_to_index.json", "w") as f:
    json.dump(phoneme_to_index, f)

print("Phoneme-to-index mapping created and saved.")

# Convert phoneme sequences to their corresponding indices
for data in tqdm(mfcc_with_phonemes, desc="Converting Phonemes to Indices"):
    phonemes = data["phonemes"]
    
    # Convert phonemes to indices using the phoneme-to-index mapping
    phoneme_indices = []
    for phoneme in phonemes:
        if phoneme in phoneme_to_index:
            phoneme_indices.append(phoneme_to_index[phoneme])
        else:
            phoneme_indices.append(phoneme_to_index.get("<UNK>", -1))  # Use -1 or <UNK> for missing phonemes
    
    # Replace the phonemes with their corresponding indices
    data["phoneme_indices"] = phoneme_indices

# Save the new data (MFCC + phoneme indices)
with open("mfcc_with_phoneme_indices.json", "w") as f:
    json.dump(mfcc_with_phonemes, f)

print("Phonemes converted to indices and saved.")

# Prepare features (MFCCs) and labels (phoneme indices)
X = []  # Features: list of MFCC sequences (each a 2D array)
lengths = []  # Needed for hmmlearn to know sequence boundaries

# Check if `mfcc_with_phonemes` is populated
print(f"mfcc_with_phonemes contains {len(mfcc_with_phonemes)} elements")

# Iterate over each data point and collect MFCCs and lengths
for data in mfcc_with_phonemes:
    mfcc = np.array(data["mfcc"])
    X.append(mfcc)
    lengths.append(len(mfcc))

# Concatenate all MFCCs into a single 2D array
X_concat = np.concatenate(X, axis=0)

# Save the prepared features and sequence lengths
np.save("mfcc_features_concat.npy", X_concat)
np.save("mfcc_lengths.npy", lengths)

print("Features and sequence lengths prepared for HMM training and saved.")

import numpy as np
import os
import json
import joblib
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.metrics import accuracy_score

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
            
            # Only add valid segments with correct feature dimension
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
            X = np.vstack(sequences)
            lengths = [len(seq) for seq in sequences]
            
            model = hmm.GaussianHMM(
                n_components=3,
                covariance_type='diag',
                n_iter=50,
                init_params="stmc",
                verbose=False
            )
            model.fit(X, lengths)
            
            # Save the trained model
            phoneme_models[phoneme_idx] = model
            joblib.dump(model, f"{models_dir}/{phoneme_idx}.pkl")
            print(f"✅ Trained model for phoneme {phoneme_idx} ({index_to_phoneme.get(phoneme_idx, '?')})")
            
        except Exception as e:
            print(f"❌ Failed for phoneme {phoneme_idx} ({index_to_phoneme.get(phoneme_idx, '?')}): {e}")
    
    print(f"✅ Trained {len(phoneme_models)} phoneme models")
    return phoneme_models, index_to_phoneme


# ================ TESTING CODE ================

class CombinedHMM:
    """
    Combines multiple HMM models into a single sequential model
    """
    def __init__(self, models):
        if not models:
            raise ValueError("No models provided to CombinedHMM")
            
        self.models = models
        self.n_features = models[0].means_.shape[1]
        self.n_components = sum(m.n_components for m in models)
        self.phoneme_boundaries = []
        
        # Initialize combined model parameters
        startprob = np.zeros(self.n_components)
        transmat = np.zeros((self.n_components, self.n_components))
        means = np.zeros((self.n_components, self.n_features))
        covars = np.zeros((self.n_components, self.n_features))
        
        offset = 0
        for i, m in enumerate(models):
            # Validate dimensions
            if m.means_.shape[1] != self.n_features:
                raise ValueError(f"Model {i} has inconsistent feature dimension: {m.means_.shape[1]} vs {self.n_features}")
                
            n = m.n_components
            
            # Keep track of which states belong to which phoneme
            self.phoneme_boundaries.append((offset, offset + n))
            
            # Copy startprob (only for the first model)
            if i == 0:
                startprob[offset:offset + n] = m.startprob_
            
            # Copy transmat within this model
            transmat[offset:offset + n, offset:offset + n] = m.transmat_
            
            # Add transition to the next model's first state
            if i < len(models) - 1:
                # Zero out the existing transition probabilities from the last state
                transmat[offset + n - 1, :] = 0.0
                # Set transition to next model's first state
                transmat[offset + n - 1, offset + n] = 1.0
            
            # Copy means and covars
            means[offset:offset + n] = m.means_
            
            # CRITICAL FIX: Handle covars properly based on their shape
            if len(m.covars_.shape) == 2:  # Shape is (n_components, n_features)
                covars[offset:offset + n] = m.covars_
            elif len(m.covars_.shape) == 3:  # Shape is (n_components, n_features, n_features)
                # Extract diagonal elements for 'diag' covariance type
                for j in range(n):
                    covars[offset + j] = np.diag(m.covars_[j])
            
            offset += n
        
        # Initialize the combined HMM
        self.model = hmm.GaussianHMM(n_components=self.n_components,
                                    covariance_type='diag',
                                    init_params="")
        
        # Normalize transition matrix rows to sum to 1
        row_sums = transmat.sum(axis=1)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:  # Only normalize non-zero rows
                transmat[i] = transmat[i] / row_sums[i]
            else:
                # For any row that sums to 0, add self-transition
                transmat[i, i] = 1.0
                
        # Ensure start probabilities sum to 1
        if startprob.sum() > 0:
            startprob = startprob / startprob.sum()
        else:
            # If all zeros, set first state to probability 1
            startprob[0] = 1.0
        
        self.model.startprob_ = startprob
        self.model.transmat_ = transmat
        self.model.means_ = means
        self.model.covars_ = covars
        
        # Verify transition matrix is valid
        row_sums_after = self.model.transmat_.sum(axis=1)
        if not np.allclose(row_sums_after, 1.0):
            print("⚠️ Warning: Some transition matrix rows still don't sum to 1")
            print(f"Row sums: {row_sums_after}")
    
    def decode(self, X):
        """Decode the sequence and return state sequence"""
        return self.model.predict(X)
    
    def map_states_to_phonemes(self, state_sequence, phoneme_indices):
        """Map HMM states back to phoneme indices"""
        phoneme_predictions = []
        
        for state in state_sequence:
            for i, (start, end) in enumerate(self.phoneme_boundaries):
                if start <= state < end:
                    phoneme_predictions.append(phoneme_indices[i])
                    break
        
        return phoneme_predictions

def load_phoneme_models(models_dir="hmm_models"):
    """Load trained HMM models from directory"""
    phoneme_models = {}
    
    for fname in os.listdir(models_dir):
        if not fname.endswith('.pkl'):
            continue
            
        try:
            idx = int(fname.split(".")[0])  # Extract phoneme index from filename
            model_path = os.path.join(models_dir, fname)
            model = joblib.load(model_path)
            phoneme_models[idx] = model
        except Exception as e:
            print(f"Error loading model {fname}: {e}")
    
    print(f"Loaded {len(phoneme_models)} phoneme models")
    return phoneme_models

def inspect_model_properties(phoneme_models):
    """Inspect properties of the loaded models to help with debugging"""
    print("\n=== MODEL INSPECTION ===")
    for idx, model in list(phoneme_models.items())[:3]:  # Just check first 3 models
        print(f"Model {idx}:")
        print(f"- n_components: {model.n_components}")
        print(f"- means_.shape: {model.means_.shape}")
        print(f"- covars_.shape: {model.covars_.shape}")
        print(f"- covariance_type: {model.covariance_type}")
        
        # Check if there are any NaN values
        if np.isnan(model.means_).any():
            print("  ⚠️ NaN values found in means")
        if np.isnan(model.covars_).any():
            print("  ⚠️ NaN values found in covars")
        
        # Verify transition matrix
        row_sums = model.transmat_.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            print(f"  ⚠️ Transition matrix rows don't sum to 1: {row_sums}")
        
        print()

def verify_and_fix_hmm_models(phoneme_models):
    """Verify HMM models and fix common issues"""
    print("Verifying and fixing HMM models...")
    fixed_models = {}
    
    for idx, model in phoneme_models.items():
        try:
            # Check and fix transition matrix
            transmat = model.transmat_.copy()
            row_sums = transmat.sum(axis=1)
            
            if not np.allclose(row_sums, 1.0):
                print(f"Fixing transition matrix for model {idx}...")
                for i in range(len(row_sums)):
                    if row_sums[i] > 0:
                        transmat[i] = transmat[i] / row_sums[i]
                    else:
                        transmat[i, i] = 1.0
                model.transmat_ = transmat
            
            # Check startprob
            if not np.isclose(model.startprob_.sum(), 1.0):
                print(f"Fixing start probabilities for model {idx}...")
                if model.startprob_.sum() > 0:
                    model.startprob_ = model.startprob_ / model.startprob_.sum()
                else:
                    model.startprob_ = np.zeros(model.n_components)
                    model.startprob_[0] = 1.0
            
            # Check for NaN values
            if np.isnan(model.means_).any() or np.isnan(model.covars_).any():
                print(f"⚠️ Model {idx} contains NaN values - skipping")
                continue
                
            fixed_models[idx] = model
            
        except Exception as e:
            print(f"Error fixing model {idx}: {e}")
    
    print(f"Fixed {len(fixed_models)} models")
    return fixed_models

def test_phoneme_recognition(mfcc_with_phonemes_path, phoneme_to_index_path, phoneme_models,
                           test_samples=30):
    """Test phoneme recognition using trained HMM models"""
    # Load test data
    with open(mfcc_with_phonemes_path, "r") as f:
        mfcc_with_phonemes = json.load(f)
    
    # Load phoneme mapping
    with open(phoneme_to_index_path, "r") as f:
        phoneme_to_index = json.load(f)
    
    index_to_phoneme = {int(v): k for k, v in phoneme_to_index.items()}
    
    # First, inspect the models
    inspect_model_properties(phoneme_models)
    
    true_labels = []
    predicted_labels = []
    
    for data in tqdm(mfcc_with_phonemes[:test_samples], desc="Testing"):
        phoneme_indices = data["phoneme_indices"]
        mfcc_seq = np.array(data["mfcc"])
        
        # Skip empty sequences
        if len(phoneme_indices) == 0 or mfcc_seq.shape[0] == 0:
            continue
            
        # Validate MFCC dimensions
        if mfcc_seq.shape[1] != 13:  # Assuming 13 MFCC coefficients
            print(f"Skipping sample with incorrect MFCC dimension: {mfcc_seq.shape}")
            continue
        
        # Filter valid models for the phoneme sequence
        valid_models = []
        valid_phonemes = []
        
        for p_idx in phoneme_indices:
            p_idx = int(p_idx)  # Ensure integer
            if p_idx in phoneme_models:
                model = phoneme_models[p_idx]
                if model.means_.shape[1] == mfcc_seq.shape[1]:
                    valid_models.append(model)
                    valid_phonemes.append(p_idx)
        
        # Skip if no valid models
        if not valid_models:
            print("No valid models for this phoneme sequence")
            continue
        
        try:
            # Create combined HMM and decode
            combined_hmm = CombinedHMM(valid_models)
            pred_states = combined_hmm.decode(mfcc_seq)
            
            # Map states back to phonemes
            pred_phonemes = combined_hmm.map_states_to_phonemes(pred_states, valid_phonemes)
            
            # Remove consecutive duplicates
            pred_phonemes_nodup = []
            for i in range(len(pred_phonemes)):
                if i == 0 or pred_phonemes[i] != pred_phonemes[i-1]:
                    pred_phonemes_nodup.append(pred_phonemes[i])
            
            # Limit prediction to match true sequence length
            pred_phonemes_trimmed = pred_phonemes_nodup[:len(valid_phonemes)]
            
            # Extend prediction if needed
            while len(pred_phonemes_trimmed) < len(valid_phonemes):
                pred_phonemes_trimmed.append(pred_phonemes_trimmed[-1] if pred_phonemes_trimmed else -1)
            
            # Add to evaluation lists
            true_labels.extend(valid_phonemes)
            predicted_labels.extend(pred_phonemes_trimmed)
            
        except Exception as e:
            print(f"Error in decoding: {str(e)}")
            # Print detailed information about the first model to help debugging
            if valid_models:
                m = valid_models[0]
                print(f"First model properties - n_components: {m.n_components}, means: {m.means_.shape}, covars: {m.covars_.shape}")
    
    # Calculate accuracy
    if true_labels and len(true_labels) == len(predicted_labels):
        acc = accuracy_score(true_labels, predicted_labels)
        print(f"Phoneme-level Accuracy: {acc * 100:.2f}%")
        
        # Print some example predictions for verification
        print("\nSample Predictions:")
        for i in range(min(10, len(true_labels))):
            t = true_labels[i]
            p = predicted_labels[i]
            print(f"True: {t} ({index_to_phoneme.get(t, '?')}), Pred: {p} ({index_to_phoneme.get(p, '?')})")
    else:
        print("No valid evaluation data")

# ================ Standalone Decoder Function ================

def decode_with_viterbi(phoneme_models, phoneme_sequence, mfcc_seq):
    """
    A standalone decoder function that can be used for individual decoding tasks
    """
    # Convert phoneme_sequence to list of integers if needed
    phoneme_sequence = [int(p) for p in phoneme_sequence]
    
    # Filter valid phoneme models
    models = []
    valid_phonemes = []
    for p in phoneme_sequence:
        if p in phoneme_models:
            model = phoneme_models[p]
            if model.means_.shape[1] == mfcc_seq.shape[1]:
                models.append(model)
                valid_phonemes.append(p)
    
    if not models:
        raise ValueError("No valid models for the phoneme sequence")
    
    # Create combined HMM
    combined_hmm = CombinedHMM(models)
    
    # Decode
    states = combined_hmm.decode(mfcc_seq)
    
    # Map states back to phonemes
    phoneme_predictions = combined_hmm.map_states_to_phonemes(states, valid_phonemes)
    
    # Remove duplicates (optional)
    reduced_predictions = []
    for i in range(len(phoneme_predictions)):
        if i == 0 or phoneme_predictions[i] != phoneme_predictions[i-1]:
            reduced_predictions.append(phoneme_predictions[i])
    
    return reduced_predictions


# Predict for a single index

def evaluate_utterance(utterance_index, mfcc_path, phoneme_map_path, phoneme_models):
    """
    Evaluate a single utterance from the dataset based on its index.
    Returns the actual and predicted phoneme sequences and accuracy.
    """
    import json
    import numpy as np

    with open(mfcc_path, "r") as f:
        mfcc_data = json.load(f)
    
    with open(phoneme_map_path, "r") as f:
        phoneme_to_index = json.load(f)
    
    index_to_phoneme = {int(v): k for k, v in phoneme_to_index.items()}
    
    if utterance_index < 0 or utterance_index >= len(mfcc_data):
        return f"Error: Index {utterance_index} is out of range. The dataset has {len(mfcc_data)} utterances.", None, None, None
    
    utterance = mfcc_data[utterance_index]
    
    try:
        phoneme_indices = [int(idx) for idx in utterance["phoneme_indices"]]
        mfcc_features = np.array(utterance["mfcc"])
        
        actual_phonemes = [index_to_phoneme.get(idx, f"Unknown-{idx}") for idx in phoneme_indices]
        
        valid_models = []
        valid_phoneme_indices = []
        
        for p_idx in phoneme_indices:
            if p_idx in phoneme_models:
                model = phoneme_models[p_idx]
                if model.means_.shape[1] == mfcc_features.shape[1]:
                    valid_models.append(model)
                    valid_phoneme_indices.append(p_idx)
        
        if not valid_models:
            return "No valid models for this phoneme sequence", None, None, None
        
        combined_hmm = CombinedHMM(valid_models)
        pred_states = combined_hmm.decode(mfcc_features)
        pred_phoneme_indices = combined_hmm.map_states_to_phonemes(pred_states, valid_phoneme_indices)
        
        reduced_predictions = []
        for i in range(len(pred_phoneme_indices)):
            if i == 0 or pred_phoneme_indices[i] != pred_phoneme_indices[i-1]:
                reduced_predictions.append(pred_phoneme_indices[i])
        
        pred_phoneme_indices_trimmed = reduced_predictions[:len(valid_phoneme_indices)]
        while len(pred_phoneme_indices_trimmed) < len(valid_phoneme_indices):
            pred_phoneme_indices_trimmed.append(
                pred_phoneme_indices_trimmed[-1] if pred_phoneme_indices_trimmed else -1
            )
        
        predicted_phonemes = [index_to_phoneme.get(idx, f"Unknown-{idx}") for idx in pred_phoneme_indices_trimmed]
        
        correct = sum(1 for i in range(len(valid_phoneme_indices)) 
                    if i < len(pred_phoneme_indices_trimmed) and valid_phoneme_indices[i] == pred_phoneme_indices_trimmed[i])
        accuracy = (correct / len(valid_phoneme_indices)) * 100 if valid_phoneme_indices else 0
        
        return actual_phonemes, predicted_phonemes, accuracy, None

    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()


# ================ MAIN EXECUTION ================

if __name__ == "__main__":
    # Paths
    mfcc_path = "mfcc_with_phoneme_indices.json"
    phoneme_map_path = "phoneme_to_index.json"
    models_dir = "hmm_models"

    # Training
    print("=== TRAINING PHONEME HMMs ===")
    phoneme_models, index_to_phoneme = train_phoneme_hmms(mfcc_path, phoneme_map_path, models_dir)

    # Testing
    print("\n=== TESTING PHONEME RECOGNITION ===")
    if not phoneme_models:
        print("No phoneme models found in memory. Loading from disk.")
        phoneme_models = load_phoneme_models(models_dir)
    
    if phoneme_models:
        test_phoneme_recognition(mfcc_path, phoneme_map_path, phoneme_models)
    else:
        print("Error: No phoneme models available for testing.")
    
    # Example of testing multiple utterances
    def evaluate_multiple_utterances(n, mfcc_path=mfcc_path, phoneme_map_path=phoneme_map_path, models_dir=models_dir):
        for i in range(n):
            actual_phonemes, predicted_phonemes, accuracy, error = evaluate_utterance(i, mfcc_path, phoneme_map_path, phoneme_models)
            
            if error:
                print(f"Error evaluating utterance {i}: {error}")
            else:
                print(f"=== Utterance {i} ===")
                print(f"Actual Phonemes : {actual_phonemes}")
                print(f"Predicted Phonemes : {predicted_phonemes}")
                print(f"Accuracy : {accuracy:.2f}%")
                print("----------------------------")

    evaluate_multiple_utterances(5)
