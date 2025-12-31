import torch
import dataset

def get_priors(gb_file):
    """
    Returns class prior probabilities.
    Note: These are invariant to window_size because they represent
    the biological frequency of states in the entire genome.
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
