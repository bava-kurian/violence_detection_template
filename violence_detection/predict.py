import torch
from torchvision import transforms
from video_dataset import VideoDataset, load_video
from models.i3d_model import get_model
import sys
import os
import numpy as np

# Default checkpoint and video
DEFAULT_CHECKPOINT = "model_epoch_10.pth"
DEFAULT_VIDEO = "C:/Users/bavak/Downloads/violence_detection_template/violence_detection/dataset/val/non_violent/NV_383.mp4"

# Accept arguments or use defaults
if len(sys.argv) == 3:
    checkpoint_path = sys.argv[1]
    video_path = sys.argv[2]
elif len(sys.argv) == 1:
    checkpoint_path = DEFAULT_CHECKPOINT
    video_path = DEFAULT_VIDEO
    print(f"No arguments specified. Using default model: {DEFAULT_CHECKPOINT}")
    print(f"Using default video: {DEFAULT_VIDEO}")
else:
    print(f"Usage: python {os.path.basename(__file__)} [model_checkpoint.pth] [video_file]")
    sys.exit(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (same as train.py)
transform = transforms.Compose([
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Load model
model = get_model().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Load and preprocess video (reuse VideoDataset logic)
# We'll use a helper to load a single video and apply transforms
frames = load_video(video_path)  # returns list of tensors (C, H, W)
frames = torch.stack([transform(frame) for frame in frames])  # shape: (T, C, H, W)
video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # shape: (1, C, T, H, W)

with torch.no_grad():
    output = model(video_tensor)
    output = output.squeeze().item()
    confidence = float(output)
    prediction = 1 if confidence > 0.5 else 0

print(f"Prediction: {'Violence' if prediction == 1 else 'Non-violence'}")
print(f"Confidence: {confidence:.4f}") 