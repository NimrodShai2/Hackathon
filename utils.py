from Bio import SeqIO
import torch


def parse_genbank(gb_file):
    """
    Parses a GenBank file to return the full sequence and the 4-class label tensor.

    Returns:
        full_sequence (str): The genomic sequence.
        labels (torch.Tensor): Tensor of shape [len(sequence)] with values 0-3.
    """
    # Read the GenBank file
    record = SeqIO.read(gb_file, "genbank")

    full_sequence = str(record.seq).upper()
    genome_length = len(full_sequence)

    # Initialize labels to 0 (Intergenic)
    labels = torch.zeros(genome_length, dtype=torch.long)

    # Iterate over features
    for feature in record.features:
        if feature.type != "CDS":
            continue

        # Get start, end, and strand
        start = int(feature.location.start)
        end = int(feature.location.end)
        strand = feature.location.strand  # 1 for forward, -1 for reverse

        # Mark internal coding (2)
        labels[start:end] = 2

        # Mark Start (1) and Stop (3)
        if strand == 1:
            # Forward Strand: Start is at the beginning (left)
            labels[start: start + 3] = 1
            labels[end - 3: end] = 3

        # Reverse Strand: Start is at the end (right)
        elif strand == -1:
            labels[end - 3: end] = 1
            labels[start: start + 3] = 3

    return full_sequence, labels