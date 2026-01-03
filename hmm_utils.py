import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import dataset
from numba import njit


def get_priors(gb_file):
    """
    Returns class prior probabilities.
    Note: These are invariant to window_size because they represent
    the biological frequency of states in the entire genome.
    Should be run only on the training genome.
    """
    print(f"Calculating class priors from: {gb_file}...")

    # We pass window_size=1 just to satisfy the constructor,
    # but it won't be used for the label calculation.
    full_dataset = dataset.GenomicDataset(gb_file, window_size=1)

    labels = full_dataset.full_labels
    counts = torch.bincount(labels, minlength=4).float()

    # Calculate Probabilities
    total = len(labels)
    priors = counts / total

    print(f"  > Counts: {counts.long().tolist()}")
    print(f"  > Priors: {priors.tolist()}")

    return priors


def get_CNN_emission_matrix(model: torch.nn.Module, dataset, priors, batch_size=1024, device='cuda'):
    """
    Generates the emission matrix for the Viterbi algorithm.

    Args:
        model: Trained DNA_CNN model.
        dataset: GenomicDataset (usually the validation set or full genome).
        priors: Tensor of shape [4] containing training set class frequencies.

    Returns:
        emission_matrix: Numpy array of shape [Genome_Length, 4].
    """
    model.eval()
    model.to(device)
    priors = priors.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    all_probs = []

    print("Generating CNN emissions...")
    with torch.inference_mode():
        for inputs, _ in tqdm(loader):
            inputs = inputs.to(device)

            # Get Logits from CNN
            logits = model(inputs)

            # Apply Softmax to get Posteriors P(State | DNA)
            probs = F.softmax(logits, dim=1)

            # Convert Posterior to Pseudo-Likelihood
            # We add epsilon to priors to avoid division by zero (though priors shouldn't be 0)
            epsilon = 1e-9
            adjusted_probs = probs / (priors + epsilon)

            # Move to CPU to save GPU memory
            all_probs.append(adjusted_probs.cpu().numpy())

    # Concatenate all batches
    # Shape: [N_samples, 4]
    emission_core = np.concatenate(all_probs, axis=0)

    # The dataset is shorter than the genome by 'window_size'
    # (The CNN cannot predict the first 100 or last 100 bases).
    # We pad these regions with a safe default: 100% probability of State 0 (Intergenic).

    genome_len = dataset.genome_length
    predicted_len = emission_core.shape[0]
    pad_len = (genome_len - predicted_len) // 2

    # Create padding: [pad_len, 4] where column 0 is 1.0, others 0.0
    padding = np.zeros((pad_len, 4))
    padding[:, 0] = 1.0  # Force Intergenic

    # Combine: [Left Pad] + [CNN Predictions] + [Right Pad]
    # Note: If genome_length is odd/even mismatch, handle remainder on the right
    right_pad_len = genome_len - predicted_len - pad_len
    right_padding = np.zeros((right_pad_len, 4))
    right_padding[:, 0] = 1.0

    full_emission_matrix = np.vstack([padding, emission_core, right_padding])

    print(f"Emission Matrix Shape: {full_emission_matrix.shape}")
    return full_emission_matrix


def get_learned_transition_matrix(gb_file):
    """
    Learns transition probabilities from the provided GenBank file.
    :param gb_file: Path to GenBank file.
    :return: Log-space transition matrix of shape [4, 4].
    0: Intergenic
    1: Start
    2: Coding
    3: Stop
    """
    print(f"Learning transitions from {gb_file}...")

    ds = dataset.GenomicDataset(gb_file, window_size=1)
    labels = ds.full_labels.numpy()

    counts = np.zeros((4, 4), dtype=np.float64)

    # Iterate and count transitions
    # labels[:-1] is "current", labels[1:] is "next"
    for current, next_state in zip(labels[:-1], labels[1:]):
        counts[current, next_state] += 1

    probs = counts / counts.sum(axis=1, keepdims=True)

    print("Raw Learned Probabilities:")
    print(probs)

    # Enforce strict biological constraints

    # Force Start(1) -> Coding (2)
    probs[1, :] = 0.0
    probs[1, 2] = 1.0

    # Force Stop(3) -> Intergenic(0)
    probs[3, :] = 0.0
    probs[3, 0] = 1.0

    # Clean up Intergenic(0) and Coding (2)
    probs[0, 2] = 0.0
    probs[0, 3] = 0.0
    # Renormalize row 0
    probs[0] = probs[0] / probs[0].sum()

    # We only allow Coding(2) -> Coding(2) or Coding(2) -> Stop(3)
    probs[2, 0] = 0.0
    probs[2, 1] = 0.0
    # Renormalize row 2
    probs[2] = probs[2] / probs[2].sum()

    print("\nConstrained Probabilities (Final):")
    print(probs)

    return np.log(probs + 1e-20)


def get_baseline_emission_probs(gb_file):
    """
    Calculates P(Nucleotide | State) by counting occurrences in training data.
    Returns a 4x4 Probability Table (States x Nucleotides).
    """
    print(f"Learning Baseline emissions from {gb_file}...")

    ds = dataset.GenomicDataset(gb_file, window_size=1)
    labels = ds.full_labels.numpy()

    # Convert sequence to integers (0-3) using the dataset's map
    # Ignore 'N' (4) for the learning phase to avoid noise
    seq_ints = [ds.stoi[base] for base in ds.full_sequence]
    seq_ints = np.array(seq_ints)

    # Count (State, Nucleotide) pairs
    # Shape: [4 States, 4 Nucleotides (A,C,G,T)]
    counts = np.zeros((4, 4), dtype=np.float64)

    # Mask to ignore 'N' or other weird characters
    valid_mask = (seq_ints < 4)

    valid_labels = labels[valid_mask]
    valid_seq = seq_ints[valid_mask]

    # Fast counting using numpy
    # We iterate 0-3 for states and 0-3 for nucleotides
    for s in range(4):
        state_mask = (valid_labels == s)
        state_seq = valid_seq[state_mask]

        for n in range(4):
            counts[s, n] = np.sum(state_seq == n)

    # Laplace Smoothing to avoid zero probabilities
    counts += 1.0

    # Normalize to get probabilities
    probs = counts / counts.sum(axis=1, keepdims=True)

    print("Baseline Emission Probs (Rows=States, Cols=ACGT):")
    print(probs)

    return probs


@njit
def _fill_emission_matrix(seq_ints, emission_probs_table, output_matrix):
    """
    The core logic: a simple, readable loop.
    Numba compiles this to machine code.
    """
    length = len(seq_ints)

    # 0=A, 1=C, 2=G, 3=T, 4=N/Unknown
    for i in range(length):
        base_idx = seq_ints[i]

        if base_idx < 4:
            output_matrix[i, 0] = emission_probs_table[0, base_idx]
            output_matrix[i, 1] = emission_probs_table[1, base_idx]
            output_matrix[i, 2] = emission_probs_table[2, base_idx]
            output_matrix[i, 3] = emission_probs_table[3, base_idx]
        else:
            # Default uniform for 'N' or unknown
            output_matrix[i, 0] = 0.25
            output_matrix[i, 1] = 0.25
            output_matrix[i, 2] = 0.25
            output_matrix[i, 3] = 0.25


def get_baseline_emission_matrix_numba(genome_seq_str, emission_probs_table):
    """
    Wrapper function to handle string conversion.
    """
    length = len(genome_seq_str)

    # Convert string to integers 0-3 for numba
    mapper = np.full(256, 4, dtype=np.int8)  # Default to 4
    mapper[ord('A')] = 0
    mapper[ord('C')] = 1
    mapper[ord('G')] = 2
    mapper[ord('T')] = 3

    # Convert string to bytes, then map to 0-4
    seq_bytes = np.frombuffer(genome_seq_str.encode('ascii'), dtype=np.uint8)
    seq_ints = mapper[seq_bytes]

    emission_matrix = np.zeros((length, 4), dtype=np.float64)
    _fill_emission_matrix(seq_ints, emission_probs_table, emission_matrix)

    return emission_matrix
