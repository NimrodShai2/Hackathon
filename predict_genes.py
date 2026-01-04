import argparse
import os
import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import dataset
import train_cnn
import hmm_utils
from HMM import HMM
from benchmark_utils import evaluate_genome


def load_best_model(best_path, device):
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {best_path}")
    payload = torch.load(best_path, map_location=device)
    window_size = payload.get('window_size', 200)
    state_dict = payload['state_dict']
    model = train_cnn.DNA_CNN(window_size=window_size)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, window_size


def extract_coding_regions(predicted_path, sequence_str):
    """Return list of (start, end, seq) for contiguous coding regions (state==2)."""
    coding_indices = np.where(predicted_path == 2)[0]
    regions = []
    if coding_indices.size == 0:
        return regions

    start = coding_indices[0]
    prev = coding_indices[0]
    for idx in coding_indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        regions.append((start, prev + 1, sequence_str[start:prev + 1]))
        start = idx
        prev = idx
    regions.append((start, prev + 1, sequence_str[start:prev + 1]))
    return regions


def save_fasta(regions, out_fasta):
    records = []
    for i, (start, end, seq) in enumerate(regions, 1):
        record_id = f"CDS_{i}_pos_{start}_{end}"
        records.append(SeqRecord(Seq(seq), id=record_id, description=""))
    SeqIO.write(records, out_fasta, "fasta")


def save_indices(regions, out_idx_path):
    with open(out_idx_path, 'w') as f:
        for start, end, _ in regions:
            f.write(f"{start}\t{end}\n")


def run_inference(gb_input, out_dir, best_model_path='best_cnn.pth', training_gb='data/ecoli_annotations.gb',
                  device=None):
    os.makedirs(out_dir, exist_ok=True)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load best model
    model, window_size = load_best_model(best_model_path, device)

    # Prepare dataset and matrices
    ds = dataset.GenomicDataset(gb_input, window_size=window_size)
    priors = hmm_utils.get_priors(training_gb)
    transition_matrix = hmm_utils.get_learned_transition_matrix(training_gb)

    # Evaluate genome (returns predicted path)
    _, _, predicted_path = evaluate_genome(
        model,
        HMM,
        ds,
        transition_matrix,
        priors,
        device=device,
    )

    # Extract coding regions
    regions = extract_coding_regions(predicted_path, ds.full_sequence)

    # Save outputs
    fasta_path = os.path.join(out_dir, 'predicted_cds.fasta')
    idx_path = os.path.join(out_dir, 'predicted_cds_indices.tsv')
    save_fasta(regions, fasta_path)
    save_indices(regions, idx_path)

    print(f"Saved {len(regions)} coding regions to:")
    print(f"  Indices: {idx_path}")
    print(f"  FASTA:   {fasta_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Neural HMM inference on a genome and extract predicted CDS.')
    parser.add_argument('--input', required=True, help='Path to GenBank file for inference')
    parser.add_argument('--out_dir', required=True, help='Output directory to save predictions')
    parser.add_argument('--best_model', default='best_cnn.pth', help='Path to best CNN checkpoint saved from benchmark')
    parser.add_argument('--training_gb', default='data/ecoli_annotations.gb',
                        help='Training GenBank used to learn priors/transitions')
    args = parser.parse_args()

    run_inference(args.input, args.out_dir, best_model_path=args.best_model, training_gb=args.training_gb)
