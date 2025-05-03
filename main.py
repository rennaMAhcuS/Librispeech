"""
Main ASR Pipeline Runner
This script runs the full ASR pipeline in order:
1. Preprocess data
2. Train phoneme HMMs
3. Evaluate/test the trained model
"""

import os
import sys

# Ensure the script's directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main functions from each module
from data_preprocess import main as preprocess_main
from training import main as train_main
from testing import main as test_main  # You can rename this if needed


def main():
    print("📦 Step 1: Preprocessing data...")
    preprocess_main()

    print("\n🎯 Step 2: Training phoneme HMMs...")
    train_main()

    print("\n🧪 Step 3: Testing and evaluation...")
    acc = test_main()

    return acc


if __name__ == "__main__":
    main()
