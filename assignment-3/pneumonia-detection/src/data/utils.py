from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from .dataset import XRayDataset
import os

def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = None,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        data_dir: Directory containing the dataset
        batch_size: Batch size for training
        val_split: Fraction of training data to use for validation
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory in data loader
    """

    # Get number of workers from environment or use default
    if num_workers is None:
        num_workers = int(os.getenv('NUM_WORKERS', '2'))

    # Create dataset
    dataset = XRayDataset(data_dir, split='train')
    
    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader