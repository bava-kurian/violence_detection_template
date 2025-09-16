import os
import shutil
import random

# Paths
dataset_root = "dataset_root"
train_dir = "data_split/train"
val_dir = "data_split/val"
split_ratio = 0.8  # 80% train, 20% val

# Create folders
for folder in ["angry","disgust","fear","happy","neutral","sad","surprise"]:
    os.makedirs(os.path.join(train_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, folder), exist_ok=True)

# Split
for folder in os.listdir(dataset_root):
    folder_path = os.path.join(dataset_root, folder)
    images = os.listdir(folder_path)
    random.shuffle(images)
    split_idx = int(len(images)*split_ratio)
    
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    for img in train_imgs:
        shutil.copy(os.path.join(folder_path, img), os.path.join(train_dir, folder, img))
    for img in val_imgs:
        shutil.copy(os.path.join(folder_path, img), os.path.join(val_dir, folder, img))

print("Dataset split into train and val.")
