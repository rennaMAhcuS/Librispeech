import os
import joblib
import json
import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from sklearn.metrics import accuracy_score


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
                raise ValueError(
                    f"Model {i} has inconsistent feature dimension: {m.means_.shape[1]} vs {self.n_features}")

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
                if i == 0 or pred_phonemes[i] != pred_phonemes[i - 1]:
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
                print(
                    f"First model properties - n_components: {m.n_components}, means: {m.means_.shape}, covars: {m.covars_.shape}")

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

        return acc * 100
    else:
        print("No valid evaluation data")
        return 0


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
        if i == 0 or phoneme_predictions[i] != phoneme_predictions[i - 1]:
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
            if i == 0 or pred_phoneme_indices[i] != pred_phoneme_indices[i - 1]:
                reduced_predictions.append(pred_phoneme_indices[i])

        pred_phoneme_indices_trimmed = reduced_predictions[:len(valid_phoneme_indices)]
        while len(pred_phoneme_indices_trimmed) < len(valid_phoneme_indices):
            pred_phoneme_indices_trimmed.append(
                pred_phoneme_indices_trimmed[-1] if pred_phoneme_indices_trimmed else -1
            )

        predicted_phonemes = [index_to_phoneme.get(idx, f"Unknown-{idx}") for idx in pred_phoneme_indices_trimmed]

        correct = sum(1 for i in range(len(valid_phoneme_indices))
                      if i < len(pred_phoneme_indices_trimmed) and valid_phoneme_indices[i] ==
                      pred_phoneme_indices_trimmed[i])
        accuracy = (correct / len(valid_phoneme_indices)) * 100 if valid_phoneme_indices else 0

        return actual_phonemes, predicted_phonemes, accuracy, None

    except Exception as e:
        import traceback
        return None, None, None, traceback.format_exc()


# ================ MAIN EXECUTION ================

def main():
    # Paths
    mfcc_path = "mfcc_with_phoneme_indices.json"
    phoneme_map_path = "phoneme_to_index.json"
    models_dir = "hmm_models"

    # ---- Load phoneme models ----
    def load_phoneme_models(models_dir):
        phoneme_models = {}
        for filename in os.listdir(models_dir):
            if filename.endswith(".pkl"):
                phoneme_idx = int(filename.split(".")[0])
                model = joblib.load(os.path.join(models_dir, filename))
                phoneme_models[phoneme_idx] = model
        return phoneme_models

    # Testing
    print("\n=== TESTING PHONEME RECOGNITION ===")
    phoneme_models = load_phoneme_models(models_dir)

    if phoneme_models:
        acc = test_phoneme_recognition(mfcc_path, phoneme_map_path, phoneme_models)
    else:
        print("Error: No phoneme models available for testing.")

    # Example of testing multiple utterances
    def evaluate_multiple_utterances(n, mfcc_path=mfcc_path, phoneme_map_path=phoneme_map_path, models_dir=models_dir):
        for i in range(n):
            actual_phonemes, predicted_phonemes, accuracy, error = evaluate_utterance(i, mfcc_path, phoneme_map_path,
                                                                                      phoneme_models)

            if error:
                print(f"Error evaluating utterance {i}: {error}")
            else:
                print(f"=== Utterance {i} ===")
                print(f"Actual Phonemes : {actual_phonemes}")
                print(f"Predicted Phonemes : {predicted_phonemes}")
                print(f"Accuracy : {accuracy:.2f}%")
                print("----------------------------")

    evaluate_multiple_utterances(5)
    return acc


if __name__ == "__main__":
    main()
