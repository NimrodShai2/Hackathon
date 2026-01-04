import numpy as np

from HMM import HMM
from hmm_utils import get_CNN_emission_matrix



def calculate_metrics(predicted_path, true_labels):
    """
    Calculates Nucleotide-level Precision and Recall for the Coding State (2).
    """
    # We only care about the "Coding" state (2) for the primary metrics [cite: 103, 104]
    # Create binary masks
    pred_coding = (predicted_path == 2)
    true_coding = (true_labels == 2)

    # True Positives: Predicted Coding AND Actually Coding
    tp = np.sum(pred_coding & true_coding)

    # False Positives: Predicted Coding BUT Actually Not
    fp = np.sum(pred_coding & ~true_coding)

    # False Negatives: Predicted Not Coding BUT Actually Coding
    fn = np.sum(~pred_coding & true_coding)

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return precision, recall


def evaluate_genome(cnn_model, HMM_Class, dataset, transition_matrix, training_priors, device='cuda'):
    """
    Runs the full Hybrid decoding pipeline and evaluates performance.

    Args:
        cnn_model: Trained PyTorch model.
        HMM_Class: Your custom class (e.g. MyHMM).
        dataset: Validation GenomicDataset.
        transition_matrix: Hard-coded or learned transition probabilities.
        training_priors: Priors from the TRAINING set.
    """
    print(f"Evaluating on genome (Length: {len(dataset.full_labels)})...")

    # 1. Generate Emission Matrix (Phase 3.1 & 3.2) [cite: 69, 75]
    # This uses the CNN + Bayes Trick
    emission_mat = get_CNN_emission_matrix(cnn_model, dataset, training_priors, device=device)

    # 2. Run HMM Decoding (Phase 3.4) [cite: 90]
    # Initialize your class with the matrices
    hmm = HMM_Class(transition_matrix, emission_mat)

    # Run Viterbi to get the optimal path
    predicted_path = hmm.run_viterbi()

    # 3. Get Ground Truth
    # Convert PyTorch tensor to Numpy for comparison
    true_labels = dataset.full_labels.numpy()

    # 4. Calculate Metrics [cite: 99]
    precision, recall = calculate_metrics(predicted_path, true_labels)

    print("-" * 30)
    print(f"Results for Coding Sequence (State 2):")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print("-" * 30)

    return precision, recall, predicted_path