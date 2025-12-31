import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GenomicDataset
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


class DNA_CNN(nn.Module):
    """Convolutional Neural Network for DNA Sequence Classification.
    Architecture inspired by common practices in genomic sequence analysis:
    - Layer 1: Conv1D with 32 filters of size 11, ReLU, MaxPool1D (kernel size 2)
    - Layer 2: Conv1D with 64 filters of size 7, ReLU, MaxPool1D (kernel size 2)
    - Fully Connected Layer to output class logits.
    """

    def __init__(self, window_size=200, num_classes=4):
        super(DNA_CNN, self).__init__()

        # Layer 1: Detect broad motifs (1 full helix turn)
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=32, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Length becomes window_size / 2
        )

        # Layer 2: Detect combinations of motifs
        self.layer2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)  # Length becomes window_size / 4
        )

        # Automatic calculation of the flat feature size
        # We divide by 4 because of the two MaxPool layers (stride 2 * stride 2)
        final_length = window_size // 4
        self.flatten_size = 64 * final_length

        self.dropout = nn.Dropout(p=0.5)

        # Final Classification Layer
        self.fc = nn.Linear(self.flatten_size, num_classes)

    def forward(self, x):
        # x input is indices [Batch, Length]

        # 1. Convert to One-Hot [Batch, Length, 5]
        x = nn.functional.one_hot(x, num_classes=5).float()

        # 2. Permute to [Batch, 5, Length] for Conv1d
        x = x.permute(0, 2, 1)

        # 3. Run Convolutions
        x = self.layer1(x)
        x = self.layer2(x)

        # 4. Flatten
        x = x.view(x.size(0), -1)

        # 5. Classify
        x = self.dropout(x)
        logits = self.fc(x)
        return logits


def get_class_weights(dataset, indices):
    """Calculate class weights based on label distribution in the dataset subset."""
    print("Calculating class weights (this takes a moment)...")
    labels = dataset.full_labels[indices]
    counts = torch.bincount(labels, minlength=4).float()
    counts = counts + 1.0
    total = len(labels)
    weights = total / (4 * counts)
    print(f"Class Counts: {counts.long().tolist()}")
    print(f"Calculated Weights: {weights.tolist()}")
    # take sqrt to reduce extreme weights
    weights = torch.sqrt(weights)
    print(f"Sqrt Weights: {weights.tolist()}")
    return weights


# The Main Execution Logic
def train_model(batch_size=128, learning_rate=0.001, epochs=5, window_size=200, gb_file='data/ecoli_annotations.gb'):
    # Settings
    BATCH_SIZE = batch_size
    LEARNING_RATE = learning_rate
    EPOCHS = epochs
    WINDOW_SIZE = window_size
    GB_FILE = gb_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    full_dataset = GenomicDataset(GB_FILE, window_size=WINDOW_SIZE)
    dataset_size = len(full_dataset)
    split_idx = int(dataset_size * 0.8)

    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, dataset_size))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    # num_workers = 2 for parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # 2. Setup Model
    model = DNA_CNN(window_size=WINDOW_SIZE).to(device)
    class_weights = get_class_weights(full_dataset, train_indices).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')
    best_model_state = None

    # 3. Training Loop
    print(f"\nStarting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Wrap the loader with tqdm
        # desc: The text to the left of the bar
        # leave=True: Keep the bar on screen after finishing
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", unit="batch")

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update the progress bar with the current loss
            # This gives you a live "Loss: 0.4521" display on the right
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        # 4. Validation (End of Epoch)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        # Optional: You can also wrap validation if it's slow, but usually not needed
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        # Print summary BELOW the progress bar
        print(f"Done Epoch {epoch + 1} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            print(f"New best model found at epoch {epoch + 1} with val loss {best_val_loss:.4f}")

    # 5. Save
    print("Saving model...")
    torch.save(best_model_state, f"models/cnn_model_window_{window_size}.pth")


if __name__ == "__main__":
    train_model(window_size=1000)
