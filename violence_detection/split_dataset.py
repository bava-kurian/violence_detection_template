import os
import shutil
import random

src_dir = 'dataset'
train_dir = os.path.join(src_dir, 'train')
val_dir = os.path.join(src_dir, 'val')
split_ratio = 0.8  # 80% train, 20% val

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in ['non_violent', 'violent']:
    src_cls = os.path.join(src_dir, cls)
    videos = os.listdir(src_cls)
    random.shuffle(videos)
    split = int(len(videos) * split_ratio)
    train_videos = videos[:split]
    val_videos = videos[split:]

    train_cls_dir = os.path.join(train_dir, cls)
    val_cls_dir = os.path.join(val_dir, cls)
    os.makedirs(train_cls_dir, exist_ok=True)
    os.makedirs(val_cls_dir, exist_ok=True)

    for v in train_videos:
        shutil.copy(os.path.join(src_cls, v), os.path.join(train_cls_dir, v))
    for v in val_videos:
        shutil.copy(os.path.join(src_cls, v), os.path.join(val_cls_dir, v))

print("Dataset split complete.")