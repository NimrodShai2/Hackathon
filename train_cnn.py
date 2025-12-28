import torch
import torch.nn as nn
import torch.optim as optim
from dataset import GenomicDataset
from torch.utils.data import DataLoader, Subset
import tqdm


def load_data(batch_size=32, train_split=0.8):
    full_dataset = GenomicDataset('data/ecoli_annotations.gb')

    dataset_size = len(full_dataset)
    split_idx = int(dataset_size * train_split)

    # Generate indices for training and validation sets
    train_indices = list(range(0, split_idx))
    val_indices = list(range(split_idx, dataset_size))

    # Create Subsets based on these indices
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader