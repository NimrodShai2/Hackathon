# main_eval.py
import os
import torch
import matplotlib.pyplot as plt
import dataset
import train_cnn
import hmm_utils
from benchmark_utils import evaluate_genome, calculate_metrics
from HMM import HMM

GB_FILE_TRAIN = 'data/ecoli_annotations.gb'
GB_FILE_VAL = 'data/test_annotations.gb'
WINDOW_SIZES = [20, 100, 200, 500, 1000]
PLOT_PATH = 'benchmark_comparison.png'
BEST_MODEL_PATH = 'best_cnn.pth'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Shared matrices/priors learned from training genome
    transition_matrix = hmm_utils.get_learned_transition_matrix(GB_FILE_TRAIN)
    priors = hmm_utils.get_priors(GB_FILE_TRAIN)

    # Baseline HMM using simple emission counts
    print("\n=== Evaluating Baseline HMM ===")
    baseline_dataset = dataset.GenomicDataset(GB_FILE_VAL, window_size=max(WINDOW_SIZES))
    baseline_probs = hmm_utils.get_baseline_emission_probs(GB_FILE_TRAIN)
    baseline_emissions = hmm_utils.get_baseline_emission_matrix(baseline_dataset.full_sequence, baseline_probs)
    hmm_baseline = HMM(transition_matrix, baseline_emissions)
    path_baseline = hmm_baseline.run_viterbi()
    true_labels = baseline_dataset.full_labels.numpy()
    p_base, r_base = calculate_metrics(path_baseline, true_labels)
    print(f"Baseline Precision: {p_base:.4f}")
    print(f"Baseline Recall:    {r_base:.4f}")

    # Evaluate CNN+HMM hybrids for each window size
    results = []
    best_f1 = -1.0
    best_entry = None
    best_state_dict = None
    best_ws = None
    for ws in WINDOW_SIZES:
        model_path = f"models/cnn_model_window_{ws}.pth"
        if not os.path.exists(model_path):
            print(f"[Skip] Model file not found: {model_path}")
            continue

        print(f"\n=== Evaluating Neural HMM (window={ws}) ===")
        val_dataset = dataset.GenomicDataset(GB_FILE_VAL, window_size=ws)
        cnn = train_cnn.DNA_CNN(window_size=ws)
        state_dict = torch.load(model_path, map_location=device)
        cnn.load_state_dict(state_dict)

        p_neural, r_neural, _ = evaluate_genome(
            cnn,
            HMM,
            val_dataset,
            transition_matrix,
            priors,
            device=device,
        )
        results.append((ws, p_neural, r_neural))

        f1 = (2 * p_neural * r_neural / (p_neural + r_neural)) if (p_neural + r_neural) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_entry = (ws, p_neural, r_neural, f1)
            best_state_dict = state_dict
            best_ws = ws

    # Summary table
    print("\n--- Coding-state Precision/Recall ---")
    print(f"baseline : P={p_base:.4f}, R={r_base:.4f}")
    for ws, p, r in results:
        print(f"ws={ws:4d}: P={p:.4f}, R={r:.4f}")

    if best_entry is not None:
        print(f"\nBest CNN (by F1): ws={best_entry[0]}, P={best_entry[1]:.4f}, R={best_entry[2]:.4f}, F1={best_entry[3]:.4f}")
        torch.save({'window_size': best_ws, 'state_dict': best_state_dict}, BEST_MODEL_PATH)
        print(f"Saved best model to {BEST_MODEL_PATH}")
    else:
        print("No CNN models evaluated; best model not saved.")

    # Plot comparisons
    labels = ['baseline'] + [str(ws) for ws, _, _ in results]
    precisions = [p_base] + [p for _, p, _ in results]
    recalls = [r_base] + [r for _, _, r in results]
    x = range(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, precisions, color='steelblue')
    axes[0].set_title('Precision')
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45)

    axes[1].bar(x, recalls, color='darkorange')
    axes[1].set_title('Recall')
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45)

    fig.suptitle('Baseline vs CNN window sizes (Coding state)')
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved comparison plot to {PLOT_PATH}")


if __name__ == '__main__':
    main()
