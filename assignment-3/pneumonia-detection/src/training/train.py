# src/training/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import logging
import sys
from tqdm import tqdm

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from model.classifier import PneumoniaClassifier
from data.utils import create_dataloaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(
    data_dir: str,
    model_dir: str,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> None:
    """
    Train the pneumonia classifier
    
    Args:
        data_dir: Directory containing the dataset
        model_dir: Directory to save model checkpoints
        num_epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for optimization
        device: Device to train on ('cuda' or 'cpu')
    """
    logger.info(f"Training on device: {device}")
    
    # Create model directory if it doesn't exist
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model and move to device
    model = PneumoniaClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data_dir, batch_size)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}:")
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                path=Path(model_dir) / 'best_model.pth',
                epoch=epoch,
                optimizer=optimizer,
                loss=val_loss,
                accuracy=val_acc
            )
            logger.info("Saved new best model checkpoint")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to save model checkpoints")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    
    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )