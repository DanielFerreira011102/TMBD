import shutil
from pathlib import Path
import random
import logging
from typing import List, Dict
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def partition_data(
    source_dir: str,
    output_base_dir: str,
    num_institutions: int = 3,
    split_ratios: Dict[str, float] = None
):
    """
    Partition the dataset among multiple institutions
    
    Args:
        source_dir: Source directory containing the original dataset
        output_base_dir: Base directory where institutional data will be stored
        num_institutions: Number of institutions to partition data for
        split_ratios: Optional dictionary defining split ratios for each institution
    """
    source_path = Path(source_dir)
    output_base_path = Path(output_base_dir)
    
    if not split_ratios:
        # Equal distribution by default
        split_ratios = {
            f"institution{i+1}": 1/num_institutions 
            for i in range(num_institutions)
        }
    
    # Validate split ratios
    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 0.001:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Process each split (train/test/val) and class (NORMAL/PNEUMONIA)
    for split in ['train', 'test', 'val']:
        split_dir = source_path / split
        if not split_dir.exists():
            logger.warning(f"Split directory {split_dir} not found, skipping")
            continue
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory {class_dir} not found, skipping")
                continue
            
            # Get all images for this class
            images = list(class_dir.glob('*.jpeg'))
            random.shuffle(images)
            
            # Calculate number of images for each institution
            total_images = len(images)
            start_idx = 0
            
            for inst_name, ratio in split_ratios.items():
                num_images = int(total_images * ratio)
                inst_images = images[start_idx:start_idx + num_images]
                start_idx += num_images
                
                # Create institution directory structure
                inst_dir = output_base_path / inst_name / split / class_name
                inst_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy images
                for img_path in inst_images:
                    shutil.copy2(img_path, inst_dir / img_path.name)
                
                logger.info(f"Copied {len(inst_images)} {class_name} images to {inst_name}/{split}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Partition dataset for federated learning")
    parser.add_argument("--source_dir", type=str, required=True,
                      help="Source directory containing the original dataset")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Base directory where institutional data will be stored")
    parser.add_argument("--num_institutions", type=int, default=3,
                      help="Number of institutions to partition data for")
    
    args = parser.parse_args()
    
    partition_data(
        source_dir=args.source_dir,
        output_base_dir=args.output_dir,
        num_institutions=args.num_institutions
    )