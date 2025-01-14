# src/model/classifier.py
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Dict, Any

class PneumoniaClassifier(nn.Module):
    """
    Neural network for pneumonia classification using transfer learning
    with ResNet18 as the backbone.
    """
    def __init__(self, num_classes: int = 2, use_pretrained_weights: bool = True):
        super().__init__()
        # Determine weights based on the argument
        weights = ResNet18_Weights.DEFAULT if use_pretrained_weights else None

        # Load ResNet18 with or without pretrained weights
        self.backbone = resnet18(weights=weights)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-2]:
            param.requires_grad = False
            
        # Modify final layer for classification
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
    
    def save_checkpoint(self, path: str, epoch: int, optimizer: torch.optim.Optimizer, 
                       loss: float, **kwargs) -> None:
        """Save model checkpoint with additional metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'loss': loss,
            **kwargs
        }
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, map_location: str = 'cpu') -> Dict[str, Any]:
        """Load model checkpoint and return model and metadata"""
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        model = cls()
        model.load_state_dict(checkpoint['model_state_dict'])
        return model, checkpoint