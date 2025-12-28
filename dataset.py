import torch
from torch.utils.data import Dataset
import utils


class GenomicDataset(Dataset):
    """
    PyTorch Dataset for genomic sequences and their coding labels.
    Each sample is a window of DNA sequence with its corresponding label (0, 1, 2, or 3).
    0: Intergenic (Background)
    1: Start Codon
    2: Coding Sequence (Internal CDS)
    3: Stop Codon
    """
    def __init__(self, gb_file, window_size=200):
        """
        Args:
            gb_file (string): Path to the .gb/.gbk file.
            window_size (int): Size of the DNA chunk.
        """
        self.window_size = window_size

        print(f"Parsing GenBank file: {gb_file}...")
        # Get both sequence and labels from the single file
        self.full_sequence, self.full_labels = utils.parse_genbank(gb_file)

        self.genome_length = len(self.full_sequence)
        self.stoi = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

    def __len__(self):
        # We slide the window by 1 base pair (stride=1) to get maximum training data.
        return self.genome_length - self.window_size

    def __getitem__(self, idx):
        """
        Retrieves one sample (a window of DNA and its corresponding label).
        """
        # Get the window of DNA sequence
        seq_window_str = self.full_sequence[idx: idx + self.window_size]

        # We use indices (e.g., A=0, T=3) instead of One-Hot encoding.
        seq_window = torch.tensor([self.stoi.get(base, 4) for base in seq_window_str], dtype=torch.long)

        # Get the label for the center base of the window, which determines the label for the entire window
        center_idx = idx + (self.window_size // 2)
        label = self.full_labels[center_idx]

        return seq_window, label