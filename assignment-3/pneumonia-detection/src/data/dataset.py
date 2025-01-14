import os
from typing import List, Tuple, Optional
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class XRayDataset(Dataset):
    """Dataset class for X-ray images"""
    def __init__(self, 
                 data_dir: str, 
                 transform: Optional[transforms.Compose] = None,
                 split: str = 'train'):
        """
        Args:
            data_dir: Root directory containing train/test/val folders
            transform: Optional transform to apply to images
            split: One of 'train', 'test', or 'val'
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or self.get_default_transforms()
        
        # Load all image paths and labels
        self.image_paths, self.labels = self._load_dataset()
        
    def _load_dataset(self) -> Tuple[List[Path], List[int]]:
        """Load dataset paths and labels"""
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        image_paths = []
        labels = []
        
        # Load NORMAL images (label 0)
        normal_dir = split_dir / 'NORMAL'
        for img_path in normal_dir.glob('*.jpeg'):
            image_paths.append(img_path)
            labels.append(0)
            
        # Load PNEUMONIA images (label 1)
        pneumonia_dir = split_dir / 'PNEUMONIA'
        for img_path in pneumonia_dir.glob('*.jpeg'):
            image_paths.append(img_path)
            labels.append(1)
            
        logger.info(f"Loaded {len(image_paths)} images for {self.split} split")
        return image_paths, labels
    
    @staticmethod
    def get_default_transforms() -> transforms.Compose:
        """Default transformations for X-ray images"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {img_path}: {str(e)}")
            # Return a zero tensor and the label if image loading fails
            return torch.zeros((3, 224, 224)), label