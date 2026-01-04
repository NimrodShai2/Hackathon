# Neural HMM Gene Finder

Hybrid pipeline that pairs a CNN with an HMM to label genomic sequences (intergenic/start/coding/stop), compare multiple window sizes, and extract predicted coding sequences (CDS) from new genomes.

## Repo layout
- `train_cnn.py` — Train a CNN for a chosen window size and save to `models/cnn_model_window_{size}.pth`.
- `benchmark.py` — Evaluate baseline HMM and CNN+HMM hybrids for window sizes 20/100/200/500/1000; pick best model (by F1), save `best_cnn.pth`, and plot `benchmark_comparison.png`.
- `predict_genes.py` — Run inference on a new GenBank file using `best_cnn.pth`; outputs predicted CDS indices (TSV) and sequences (FASTA).
- `dataset.py` / `utils.py` — GenBank parsing and dataset definition.
- `hmm_utils.py` — Priors, transitions, emissions helpers.
- `HMM.py` — Viterbi decoder.

## Setup
```bash
pip install -r requirements.txt
```
Torch with GPU is recommended but CPU works (slower).

## Train a CNN (optional if models already exist)
```bash
python train_cnn.py --help  # edit defaults in script if needed
python train_cnn.py         # uses window_size=1000 by default
```

## Benchmark models and pick the best
Runs baseline HMM plus CNN+HMM for window sizes 20/100/200/500/1000, saves best to `best_cnn.pth`, and writes `benchmark_comparison.png`.
```bash
python benchmark.py
```

## Predict genes on a new genome
Uses the saved `best_cnn.pth` to label a new GenBank file and extract CDS.
```bash
python predict_genes.py --input path/to/genome.gb --out_dir outputs \
  --best_model best_cnn.pth --training_gb data/ecoli_annotations.gb
```
Outputs:
- `outputs/predicted_cds_indices.tsv` — tab-separated start/end indices (0-based, end-exclusive).
- `outputs/predicted_cds.fasta` — FASTA of predicted CDS sequences.

## Notes
- Transition/prior stats default to `data/ecoli_annotations.gb`; override via `--training_gb` if using another training genome.
- Model checkpoints must match their window size (e.g., `cnn_model_window_200.pth`).
- GenBank input is required (for sequence and annotations).

