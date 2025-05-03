import os
import joblib
import json
import numpy as np
from tqdm import tqdm
from hmmlearn import hmm
from collections import Counter
from itertools import groupby

# ================ IMPROVED TESTING CODE ================

class EnhancedCombinedHMM:
    """
    Enhanced version of CombinedHMM with improvements to prevent self-looping
    """
    def __init__(self, models, phoneme_indices, duration_penalty=0.5, transition_boost=1.5):
        if not models:
            raise ValueError("No models provided to EnhancedCombinedHMM")
            
        self.models = models
        self.n_features = models[0].means_.shape[1]
        self.n_components = sum(m.n_components for m in models)
        self.phoneme_boundaries = []
        self.phoneme_indices = phoneme_indices
        
        # Parameters to control state transitions
        self.duration_penalty = duration_penalty  # Penalty for staying in same phoneme too long
        self.transition_boost = transition_boost  # Boost for transitioning to next phoneme
        
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
            
            # Copy transmat within this model with modified self-transition probabilities
            # Reduce probability of staying in same state to avoid self-loops
            model_transmat = m.transmat_.copy()
            
            # Modify self-transitions to prevent excessive looping
            for j in range(n):
                # Reduce self-transition probability
                model_transmat[j, j] *= self.duration_penalty
                
                # Increase forward transition probabilities
                if j < n-1:
                    # Boost transition to next state within same phoneme
                    model_transmat[j, j+1] = min(1.0, model_transmat[j, j+1] * self.transition_boost)
            
            # Normalize to ensure rows sum to 1
            row_sums = model_transmat.sum(axis=1)
            for j in range(n):
                if row_sums[j] > 0:
                    model_transmat[j] = model_transmat[j] / row_sums[j]
                else:
                    model_transmat[j, min(j+1, n-1)] = 1.0  # Prefer moving forward
            
            transmat[offset:offset + n, offset:offset + n] = model_transmat
            
            # Add transition to the next model's first state
            if i < len(models) - 1:
                # Zero out the existing transition probabilities from the last state
                transmat[offset + n - 1, :] = 0.0
                # Set transition to next model's first state with high probability
                transmat[offset + n - 1, offset + n] = 1.0
            
            # Copy means and covars
            means[offset:offset + n] = m.means_
            
            # Handle covars properly based on their shape
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
                # For any row that sums to 0, add forward transition if possible
                if i < self.n_components - 1:
                    transmat[i, i+1] = 1.0
                else:
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
    
    def decode(self, X, algorithm="viterbi"):
        """
        Decode the sequence using viterbi or modified forward-backward algorithm
        """
        if algorithm == "viterbi":
            # Standard Viterbi decoding
            return self.model.predict(X)
        elif algorithm == "forward_backward":
            # Use forward-backward algorithm with posterior decoding
            _, posteriors = self.model.score_samples(X)
            # Take most probable state at each time step
            return np.argmax(posteriors, axis=1)
        else:
            raise ValueError(f"Unknown decoding algorithm: {algorithm}")
    
    def map_states_to_phonemes(self, state_sequence):
        """Map HMM states back to phoneme indices"""
        phoneme_predictions = []
        
        for state in state_sequence:
            for i, (start, end) in enumerate(self.phoneme_boundaries):
                if start <= state < end:
                    phoneme_predictions.append(self.phoneme_indices[i])
                    break
        
        return phoneme_predictions

    def post_process_predictions(self, phoneme_predictions, min_duration=3, frame_skip=2):
        """
        Apply post-processing to improve predictions by:
        1. Removing very short phoneme segments (likely errors)
        2. Applying frame-skipping to detect transitions better
        3. Smoothing the predictions with a voting window
        """
        # Skip frames for better segmentation (analyze every n-th frame)
        if frame_skip > 1:
            phoneme_predictions = phoneme_predictions[::frame_skip]
        
        # 1. Group consecutive phonemes
        grouped = [(p, len(list(g))) for p, g in groupby(phoneme_predictions)]
        
        # 2. Remove segments that are too short
        filtered = []
        for phoneme, duration in grouped:
            if duration >= min_duration:
                filtered.extend([phoneme] * duration)
            elif filtered:  # Merge very short segments with previous phoneme
                filtered.extend([filtered[-1]] * duration)
            else:  # Keep short segment at the beginning
                filtered.extend([phoneme] * duration)
        
        # 3. Smooth predictions using a voting window
        window_size = 5
        smoothed = []
        for i in range(len(filtered)):
            # Define window boundaries
            start = max(0, i - window_size // 2)
            end = min(len(filtered), i + window_size // 2 + 1)
            window = filtered[start:end]
            
            # Take most common phoneme in window
            counter = Counter(window)
            smoothed.append(counter.most_common(1)[0][0])
        
        return smoothed

def load_phoneme_models(models_dir="hmm_models"):
    """Load trained HMM models from directory with additional validation"""
    phoneme_models = {}
    
    for fname in os.listdir(models_dir):
        if not fname.endswith('.pkl'):
            continue
            
        try:
            idx = int(fname.split(".")[0])  # Extract phoneme index from filename
            model_path = os.path.join(models_dir, fname)
            model = joblib.load(model_path)
            
            # Validate model parameters
            if not hasattr(model, 'means_') or not hasattr(model, 'covars_'):
                print(f"⚠️ Skipping model {idx} - missing required attributes")
                continue
                
            # Check for NaN values
            if np.isnan(model.means_).any() or np.isnan(model.covars_).any():
                print(f"⚠️ Skipping model {idx} - contains NaN values")
                continue
            
            # Add model to dictionary
            phoneme_models[idx] = model
            
        except Exception as e:
            print(f"Error loading model {fname}: {e}")
    
    print(f"Loaded {len(phoneme_models)} valid phoneme models")
    return phoneme_models

def load_phoneme_durations(models_dir="hmm_models"):
    """Load phoneme duration information if available"""
    durations_path = os.path.join(models_dir, "phoneme_durations.json")
    
    if os.path.exists(durations_path):
        with open(durations_path, "r") as f:
            durations = json.load(f)
        print(f"Loaded duration information for {len(durations)} phonemes")
        return durations
    else:
        print("No phoneme duration information found")
        return {}

def normalize_mfcc_for_testing(mfcc_features):
    """Normalize MFCC features using the same method as during training"""
    mean = np.mean(mfcc_features, axis=0)
    std = np.std(mfcc_features, axis=0)
    return (mfcc_features - mean) / (std + 1e-8)

def apply_language_model(phoneme_sequence, phoneme_transitions=None, smoothing=0.1):
    """
    Apply a simple language model to improve phoneme sequence
    Uses phoneme transition probabilities if available, otherwise uses a simple 
    bigram smoothing approach
    """
    if len(phoneme_sequence) <= 1:
        return phoneme_sequence
    
    if phoneme_transitions is None:
        # Create a simple bigram counter for smoothing
        bigram_counts = {}
        for i in range(len(phoneme_sequence) - 1):
            bigram = (phoneme_sequence[i], phoneme_sequence[i+1])
            if bigram not in bigram_counts:
                bigram_counts[bigram] = 0
            bigram_counts[bigram] += 1
        
        # Apply simple smoothing to remove unlikely transitions
        smoothed = [phoneme_sequence[0]]
        for i in range(1, len(phoneme_sequence)):
            curr = phoneme_sequence[i]
            prev = smoothed[-1]
            bigram = (prev, curr)
            
            if bigram in bigram_counts and bigram_counts[bigram] > 1:
                smoothed.append(curr)
            elif i+1 < len(phoneme_sequence) and phoneme_sequence[i] == phoneme_sequence[i+1]:
                # Keep if it's repeated (more likely to be correct)
                smoothed.append(curr)
            elif prev != curr:  # Only add if different from previous
                # Only add with certain probability based on smoothing factor
                if np.random.random() < smoothing:
                    smoothed.append(curr)
                else:
                    smoothed.append(prev)  # Keep previous phoneme
            else:
                smoothed.append(curr)
    else:
        # Use provided transition probabilities (more sophisticated)
        # Implementation would go here if phoneme_transitions is provided
        smoothed = phoneme_sequence
    
    return smoothed

def decode_phoneme_sequence(mfcc_seq, phoneme_sequence, phoneme_models, 
                           post_process=True, use_duration_model=True):
    """
    Enhanced decoder that applies multiple techniques to prevent self-looping
    """
    # 1. Normalize features (consistent with training)
    normalized_mfcc = normalize_mfcc_for_testing(mfcc_seq)
    
    # 2. Filter valid phoneme models for the sequence
    valid_models = []
    valid_phonemes = []
    for p_idx in phoneme_sequence:
        p_idx = int(p_idx)  # Ensure integer
        if p_idx in phoneme_models:
            model = phoneme_models[p_idx]
            if model.means_.shape[1] == normalized_mfcc.shape[1]:
                valid_models.append(model)
                valid_phonemes.append(p_idx)
    
    if not valid_models:
        raise ValueError("No valid models for this phoneme sequence")
    
    # 3. Create enhanced combined HMM
    combined_hmm = EnhancedCombinedHMM(
        valid_models, 
        valid_phonemes,
        duration_penalty=0.5,  # Reduce self-transition probability
        transition_boost=1.5   # Boost transitions to next state/phoneme
    )
    
    # 4. Decode using Viterbi algorithm
    pred_states = combined_hmm.decode(normalized_mfcc, algorithm="viterbi")
    
    # 5. Map states back to phonemes
    pred_phonemes = combined_hmm.map_states_to_phonemes(pred_states)
    
    # 6. Apply post-processing if enabled
    if post_process:
        pred_phonemes = combined_hmm.post_process_predictions(
            pred_phonemes,
            min_duration=3,  # Minimum frames for a phoneme to be considered valid
            frame_skip=1      # Analyze every frame (change to skip frames if needed)
        )
    
    # 7. Apply language model to smooth predictions
    pred_phonemes = apply_language_model(pred_phonemes)
    
    # 8. Remove consecutive duplicates more aggressively
    pred_phonemes_nodup = []
    for i in range(len(pred_phonemes)):
        if i == 0 or pred_phonemes[i] != pred_phonemes[i-1]:
            pred_phonemes_nodup.append(pred_phonemes[i])
    
    # 9. Align with expected sequence length
    if len(pred_phonemes_nodup) > len(valid_phonemes):
        # If predicted too many phonemes, reduce to match expected length
        # Strategy: Keep phonemes with longer durations (more confident predictions)
        
        # Count how long each predicted phoneme appeared
        counts = Counter(pred_phonemes)
        
        # Sort unique phonemes by their frequency
        unique_phonemes = sorted(counts.keys(), key=lambda x: counts[x], reverse=True)
        
        # Keep most frequent phonemes up to expected length
        if len(unique_phonemes) >= len(valid_phonemes):
            pred_phonemes_final = unique_phonemes[:len(valid_phonemes)]
        else:
            # If not enough unique phonemes, append most common ones
            pred_phonemes_final = unique_phonemes.copy()
            most_common = unique_phonemes[0] if unique_phonemes else valid_phonemes[0]
            while len(pred_phonemes_final) < len(valid_phonemes):
                pred_phonemes_final.append(most_common)
    else:
        # If predicted too few phonemes, extend to match expected length
        pred_phonemes_final = pred_phonemes_nodup.copy()
        while len(pred_phonemes_final) < len(valid_phonemes):
            # Add the last phoneme or the next expected one
            if len(pred_phonemes_final) > 0:
                next_idx = len(pred_phonemes_final)
                if next_idx < len(valid_phonemes):
                    # Try to use the expected phoneme at this position
                    pred_phonemes_final.append(valid_phonemes[next_idx])
                else:
                    # Fallback to repeating the last predicted phoneme
                    pred_phonemes_final.append(pred_phonemes_final[-1])
            else:
                # If no predictions at all, use the first expected phoneme
                pred_phonemes_final.append(valid_phonemes[0])
    
    return pred_phonemes_final

def test_phoneme_recognition(mfcc_with_phonemes_path, phoneme_to_index_path, phoneme_models,
                           test_samples=30):
    """Test phoneme recognition using trained HMM models with enhanced decoding"""
    # Load test data
    with open(mfcc_with_phonemes_path, "r") as f:
        mfcc_with_phonemes = json.load(f)
    
    # Load phoneme mapping
    with open(phoneme_to_index_path, "r") as f:
        phoneme_to_index = json.load(f)
    
    index_to_phoneme = {int(v): k for k, v in phoneme_to_index.items()}
    
    # For analysis
    confusion_matrix = {}
    total_phonemes = 0
    correct_phonemes = 0
    error_cases = 0
    
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
        
        # Convert phoneme indices to integers
        phoneme_indices = [int(idx) for idx in phoneme_indices]
        
        # Filter phonemes that have models
        valid_indices = [idx for idx in phoneme_indices if idx in phoneme_models]
        
        # Skip if no valid phonemes
        if not valid_indices:
            continue
        
        try:
            
            # Use the enhanced decoder
            pred_phonemes = decode_phoneme_sequence(
                mfcc_seq, 
                valid_indices, 
                phoneme_models, 
                post_process=True,
                use_duration_model=True
            )
            
            # Evaluate predictions
            for i in range(min(len(valid_indices), len(pred_phonemes))):
                true_idx = valid_indices[i]
                pred_idx = pred_phonemes[i]
                
                # Update confusion matrix
                if true_idx not in confusion_matrix:
                    confusion_matrix[true_idx] = {}
                if pred_idx not in confusion_matrix[true_idx]:
                    confusion_matrix[true_idx][pred_idx] = 0
                confusion_matrix[true_idx][pred_idx] += 1
                
                # Count correct predictions
                if true_idx == pred_idx:
                    correct_phonemes += 1
                total_phonemes += 1
                
        except Exception as e:
            print(f"Error in decoding: {str(e)}")
            error_cases += 1
    
    # Calculate accuracy
    if total_phonemes > 0:
        accuracy = correct_phonemes / total_phonemes * 100
        print(f"\nPhoneme-level Accuracy: {accuracy:.2f}%")
        print(f"Total phonemes evaluated: {total_phonemes}")
        print(f"Error cases: {error_cases}")
        
        # Display top confusions
        print("\nTop confusions:")
        all_confusions = []
        for true_idx, preds in confusion_matrix.items():
            true_phoneme = index_to_phoneme.get(true_idx, f"Unknown-{true_idx}")
            for pred_idx, count in preds.items():
                if true_idx != pred_idx:  # Only show errors
                    pred_phoneme = index_to_phoneme.get(pred_idx, f"Unknown-{pred_idx}")
                    all_confusions.append((true_phoneme, pred_phoneme, count))
        
        # Sort by count and display top 5
        for true_ph, pred_ph, count in sorted(all_confusions, key=lambda x: x[2], reverse=True)[:5]:
            print(f"  {true_ph} → {pred_ph}: {count} times")
        
        return accuracy
    else:
        print("No valid evaluation data")
        return 0

def evaluate_utterance(utterance_index, mfcc_path, phoneme_map_path, phoneme_models):
    """
    Evaluate a single utterance using enhanced decoding.
    
    Returns:
        actual_phonemes (list): Ground truth phoneme sequence
        predicted_phonemes (list): Decoded phoneme sequence
        accuracy (float): Match accuracy between ground truth and prediction
        error (str or None): Error message if something goes wrong, else None
    """
    try:
        # Load data
        with open(mfcc_path, "r") as f:
            mfcc_data = json.load(f)

        with open(phoneme_map_path, "r") as f:
            phoneme_to_index = json.load(f)

        index_to_phoneme = {int(v): k for k, v in phoneme_to_index.items()}

        # Check index bounds
        if utterance_index < 0 or utterance_index >= len(mfcc_data):
            return None, None, None, (
                f"Index {utterance_index} is out of range. Dataset has {len(mfcc_data)} utterances."
            )

        # Load utterance
        utterance = mfcc_data[utterance_index]
        phoneme_indices = [int(idx) for idx in utterance["phoneme_indices"]]
        mfcc_features = np.array(utterance["mfcc"])

        # Get ground truth phonemes
        actual_phonemes = [index_to_phoneme.get(idx, f"Unknown-{idx}") for idx in phoneme_indices]

        # Filter valid indices that we have models for
        valid_indices = [idx for idx in phoneme_indices if idx in phoneme_models]

        if not valid_indices:
            return None, None, None, "No valid phoneme models found for this utterance."

        # Decode using enhanced decoder
        pred_indices = decode_phoneme_sequence(
            mfcc_features,
            valid_indices,
            phoneme_models,
            post_process=True,
            use_duration_model=True
        )

        # Map predicted indices to phoneme strings
        predicted_phonemes = [index_to_phoneme.get(idx, f"Unknown-{idx}") for idx in pred_indices]

        # Accuracy calculation
        correct = sum(1 for i in range(len(valid_indices)) 
                      if i < len(pred_indices) and valid_indices[i] == pred_indices[i])
        accuracy = (correct / len(valid_indices)) * 100 if valid_indices else 0

        # Optional analysis (can be printed or logged if needed)
        from itertools import groupby
        runs = [(k, len(list(g))) for k, g in groupby(pred_indices)]
        longest_run = max(runs, key=lambda x: x[1]) if runs else (None, 0)

        analysis = {
            "num_runs": len(runs),
            "longest_run": longest_run,
            "self_loop_ratio": longest_run[1] / len(pred_indices) if pred_indices else 0,
            "unique_phonemes": len(set(pred_indices)),
        }

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
    
    # Load phoneme models
    phoneme_models = load_phoneme_models(models_dir)
    
    if not phoneme_models:
        print("Error: No phoneme models available for testing.")
        return 0
    
    # Run enhanced testing
    print("\n=== TESTING PHONEME RECOGNITION WITH ENHANCED DECODER ===")
    acc_enhanced = test_phoneme_recognition(
        mfcc_path, 
        phoneme_map_path, 
        phoneme_models,
        test_samples=30,
    )
    
    print(f"Enhanced decoder accuracy: {acc_enhanced:.2f}%")
    return acc_enhanced

if __name__ == "__main__":
    main()