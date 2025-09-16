import os
import shutil
import random

src_dir = os.path.join(os.path.dirname(__file__), 'Dataset')
train_dir = os.path.join(src_dir, 'train')
val_dir = os.path.join(src_dir, 'val')
test_dir = os.path.join(src_dir, 'test')

train_ratio = 0.7  # 70% train
val_ratio = 0.15   # 15% val
test_ratio = 0.15  # 15% test

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for cls in ['NonViolence', 'Violence']:
    src_cls = os.path.join(src_dir, cls)
    videos = os.listdir(src_cls)
    random.shuffle(videos)
    total = len(videos)
    train_split = int(total * train_ratio)
    val_split = int(total * val_ratio)
    
    train_videos = videos[:train_split]
    val_videos = videos[train_split:train_split + val_split]
    test_videos = videos[train_split + val_split:]

    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)
    test_cls_dir = os.path.join(test_dir, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)
    os.makedirs(test_cls_dir, exist_ok=True)

    for v in train_videos:
        shutil.copy(os.path.join(src_cls, v), os.path.join(train_cls_dir, v))
    for v in val_videos:
        shutil.copy(os.path.join(src_cls, v), os.path.join(val_cls_dir, v))
    for v in test_videos:
        shutil.copy(os.path.join(src_cls, v), os.path.join(test_cls_dir, v))

print("Dataset split into train, val, and test sets.")