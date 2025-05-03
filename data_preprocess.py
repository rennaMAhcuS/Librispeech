# === Import Libraries ===
import os
import requests
import tarfile
import shutil
import json
import random
import re
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
from tqdm import tqdm
from glob import glob
from scipy.io.wavfile import read
from python_speech_features import mfcc
from g2p_en import G2p
import nltk

# Drive Link
# https://drive.google.com/drive/folders/1WRnzyVkOxhiHAfUU_cUnUyidkpEFoWpW?usp=sharing

# === NLTK Setup ===
nltk.data.path.append('nltk_data')
nltk.download('punkt', download_dir='nltk_data')
nltk.download('averaged_perceptron_tagger_eng', download_dir='nltk_data')
nltk.download('cmudict', download_dir='nltk_data')  # <-- For CMU Pronouncing Dictionary

from nltk.corpus import cmudict

# === Constants ===
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
TAR_PATH = "train-clean-100.tar.gz"
EXTRACT_DIR = "audio_files"
SOURCE_ROOT = os.path.join(EXTRACT_DIR, "LibriSpeech/train-clean-100")
WAV_ROOT = "train-clean-100-wav"
SELECTED_JSON = "selected_files.json"

# === File Download ===
def download_file(url, dest):
    if not os.path.exists(dest):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(dest, 'wb') as f:
                for chunk in tqdm(r.iter_content(1024), total=total // 1024, unit='KB', desc="Downloading"):
                    f.write(chunk)

# === Extraction ===
def extract_tar(tar_path, dest_dir):
    if not os.path.exists(dest_dir):
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=dest_dir)
        print(f"Extracted to: {dest_dir}")
    else:
        print(f"Already extracted: {dest_dir}")

# === FLAC to WAV Conversion ===
def convert_flac_to_wav(source_root, wav_root):
    flac_files = glob(os.path.join(source_root, "**/*.flac"), recursive=True)
    for flac_path in tqdm(flac_files, desc="Converting FLAC to WAV"):
        rel_path = os.path.relpath(flac_path, source_root).replace('.flac', '.wav')
        wav_path = os.path.join(wav_root, rel_path)
        os.makedirs(os.path.dirname(wav_path), exist_ok=True)
        audio, sr = sf.read(flac_path)
        sf.write(wav_path, audio, sr)

# === Copy Transcripts ===
def copy_transcripts(source_root, wav_root):
    for root, _, files in os.walk(source_root):
        for file in files:
            if file.endswith(".trans.txt"):
                rel_dir = os.path.relpath(root, source_root)
                dest_dir = os.path.join(wav_root, rel_dir)
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy(os.path.join(root, file), os.path.join(dest_dir, file))

# === Extract MFCC Features ===
def extract_mfcc(wav_path, num_mfcc=13):
    sr, signal = read(wav_path)
    return mfcc(signal, sr, numcep=num_mfcc)

# === Visualization (optional) ===
def visualize_waveform_and_mfcc(wav_path):
    signal, sr = librosa.load(wav_path, sr=None)
    mfccs = librosa.feature.mfcc(signal, sr=sr, n_mfcc=13)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(signal)
    plt.title("Waveform")
    plt.subplot(1, 2, 2)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title("MFCCs")
    plt.tight_layout()
    plt.show()

# === Select ~1 hour of audio ===
def select_files(wav_files, target_sec=3600):
    random.shuffle(wav_files)
    selected = []
    duration = 0
    for path in wav_files:
        signal, sr = sf.read(path)
        duration += len(signal) / sr
        selected.append(path)
        if duration >= target_sec:
            break
    with open(SELECTED_JSON, 'w') as f:
        json.dump(selected, f)
    return selected

# === Clean Transcript ===
def clean_transcript(text):
    return re.sub(r"[^a-z\s]", "", text.lower())

# === Get Transcript for WAV ===
def get_transcript(wav_path):
    parts = wav_path.split(os.sep)
    speaker, chapter = parts[1], parts[2]
    transcript_path = os.path.join(WAV_ROOT, speaker, chapter, f"{speaker}-{chapter}.trans.txt")
    audio_id = os.path.basename(wav_path).replace(".wav", "")
    try:
        with open(transcript_path, "r") as f:
            for line in f:
                if audio_id in line:
                    return line.split(" ", 1)[1].strip()
    except FileNotFoundError:
        return None
    return None

# === Convert Text to Phonemes ===
g2p = G2p()

def transcript_to_phonemes(text):
    words = clean_transcript(text).split()
    phonemes = []
    for word in words:
        phonemes += [re.sub(r'\d', '', p) for p in g2p(word)]
    return phonemes

# === Map MFCCs to Phonemes ===
def build_dataset(wav_paths):
    data = []
    for wav_path in tqdm(wav_paths, desc="Processing files"):
        try:
            signal, sr = sf.read(wav_path)
            if signal.ndim != 1: continue
            mfcc_feat = mfcc(signal, sr, numcep=13)
            if mfcc_feat.ndim != 2: continue
            transcript = get_transcript(wav_path)
            if not transcript: continue
            phonemes = transcript_to_phonemes(transcript)
            data.append({
                "path": wav_path,
                "mfcc": mfcc_feat.tolist(),
                "phonemes": phonemes
            })
        except Exception as e:
            print(f"Error with {wav_path}: {e}")
    return data

# === Create Phoneme Index Map ===
def create_phoneme_index_map(dataset):
    phonemes = set(p for d in dataset for p in d["phonemes"])
    return {p: i for i, p in enumerate(sorted(phonemes))}

# === Encode Phonemes to Indices ===
def encode_phonemes(dataset, phoneme_to_idx):
    for d in dataset:
        d["phoneme_indices"] = [phoneme_to_idx.get(p, -1) for p in d["phonemes"]]
    return dataset

# === Save Feature Arrays ===
def save_for_training(dataset):
    mfcc_seqs = [np.array(d["mfcc"]) for d in dataset]
    lengths = [len(m) for m in mfcc_seqs]
    X = np.concatenate(mfcc_seqs, axis=0)
    np.save("mfcc_features_concat.npy", X)
    np.save("mfcc_lengths.npy", lengths)

# === Create Inverse Phoneme Dictionary ===
def create_inverse_phoneme_dict():
    cmu = cmudict.dict()
    inverse_dict = {}
    for word, phoneme_lists in cmu.items():
        for phonemes in phoneme_lists:
            clean_phonemes = [re.sub(r'\d', '', p) for p in phonemes]
            key = " ".join(clean_phonemes)
            if key not in inverse_dict:
                inverse_dict[key] = word.lower()
    with open("inverse_phoneme_dict.json", "w") as f:
        json.dump(inverse_dict, f)
    print(f"Saved inverse dictionary with {len(inverse_dict)} entries.")

# === Pipeline Execution ===
def main():
    download_file(LIBRISPEECH_URL, TAR_PATH)
    extract_tar(TAR_PATH, EXTRACT_DIR)

    if not os.path.exists(WAV_ROOT):
        convert_flac_to_wav(SOURCE_ROOT, WAV_ROOT)
        copy_transcripts(SOURCE_ROOT, WAV_ROOT)

    wav_files = glob(os.path.join(WAV_ROOT, "**/*.wav"), recursive=True)
    selected = select_files(wav_files)

    dataset = build_dataset(selected)
    with open("mfcc_with_phonemes.json", "w") as f:
        json.dump(dataset, f)

    phoneme_to_idx = create_phoneme_index_map(dataset)
    with open("phoneme_to_index.json", "w") as f:
        json.dump(phoneme_to_idx, f)

    dataset = encode_phonemes(dataset, phoneme_to_idx)
    with open("mfcc_with_phoneme_indices.json", "w") as f:
        json.dump(dataset, f)

    save_for_training(dataset)

    create_inverse_phoneme_dict()
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
