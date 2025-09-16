import os
import shutil
import random
from pathlib import Path

def setup_folders():
    """Create the folder structure for train/val splits"""
    base_folders = ['train', 'val']
    class_folders = ['violence', 'non_violence']
    
    # Create main data folder if it doesn't exist
    dataset_path = Path('dataset')
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    # Create train/val folders with violence/non_violence subfolders
    for base in base_folders:
        for cls in class_folders:
            os.makedirs(dataset_path / base / cls, exist_ok=True)
    
    print("Created folder structure:")
    print("\n".join(str(p) for p in dataset_path.rglob("*") if p.is_dir()))
    return dataset_path

def split_dataset(source_path, split_ratio=0.8):
    """
    Split the dataset into train/val sets
    Args:
        source_path: Path to source data folder containing violence/non_violence folders
        split_ratio: Float between 0 and 1, representing the train set ratio
    """
    assert 0 < split_ratio < 1, "Split ratio must be between 0 and 1"
    
    # Setup folders
    dataset_path = setup_folders()
    splits = ['train', 'val']
    
    # Process each class folder
    for class_folder in ['violence', 'non_violence']:
        source_class_path = Path(source_path) / class_folder
        if not source_class_path.exists():
            raise RuntimeError(f"Source folder not found: {source_class_path}")
        
        # Get all audio files
        files = [f for f in source_class_path.glob("*") 
                if f.suffix.lower() in ('.wav', '.mp3', '.mp4')]
        random.shuffle(files)
        
        # Calculate split sizes
        total = len(files)
        train_size = int(total * split_ratio)
        
        # Split files
        train_files = files[:train_size]
        val_files = files[train_size:]
        
        # Copy files to respective folders
        for split, split_files in zip(splits, [train_files, val_files]):
            dest_folder = dataset_path / split / class_folder
            for file in split_files:
                shutil.copy2(file, dest_folder / file.name)
            print(f"{split.capitalize()} {class_folder}: {len(split_files)} files")

if __name__ == "__main__":
    # Source data folder containing violence/non_violence folders
    SOURCE_PATH = "data"
    split_dataset(SOURCE_PATH, split_ratio=0.8)  # 80% train, 20% validation
    print("\nDataset split completed successfully!")