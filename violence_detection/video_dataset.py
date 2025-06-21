import torch
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform
        self.classes = ['non_violent', 'violent']
        self.data = []

        for label, cls in enumerate(self.classes):
            class_folder = os.path.join(root_dir, cls)
            for video in os.listdir(class_folder):
                self.data.append((os.path.join(class_folder, video), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        frames = self.load_video(video_path)
        if self.transform:
            frames = torch.stack([self.transform(img) for img in frames])
        return frames.permute(1, 0, 2, 3), torch.tensor(label, dtype=torch.long)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames).astype(int)

        for i in range(total_frames):
            ret, frame = cap.read()
            if i in frame_indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(torch.tensor(frame / 255.0, dtype=torch.float32).permute(2, 0, 1))
        cap.release()
        if len(frames) < self.num_frames:
            frames += [frames[-1]] * (self.num_frames - len(frames))
        return frames[:self.num_frames]