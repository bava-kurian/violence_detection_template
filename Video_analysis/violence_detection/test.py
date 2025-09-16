import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from video_dataset import VideoDataset
from models.i3d_model import get_model
import sys
import os

# Default checkpoint
DEFAULT_CHECKPOINT = "model_epoch_10.pth"

# Accept checkpoint as argument, else use default
if len(sys.argv) == 2:
    checkpoint_path = sys.argv[1]
elif len(sys.argv) == 1:
    checkpoint_path = DEFAULT_CHECKPOINT
    print(f"No checkpoint specified. Using default: {DEFAULT_CHECKPOINT}")
else:
    print(f"Usage: python {os.path.basename(__file__)} [model_checkpoint.pth]")
    sys.exit(1)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform (same as train.py)
transform = transforms.Compose([
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

# Test dataset and loader
test_set = VideoDataset('violence_detection/Dataset/test', transform=transform)
test_loader = DataLoader(test_set, batch_size=4)

# Model
model = get_model().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

criterion = nn.BCELoss()

total_loss = 0
correct = 0
total = 0

print("Starting testing...")
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.numel()

accuracy = 100 * correct / total if total > 0 else 0
avg_loss = total_loss / len(test_loader) if len(test_loader) > 0 else 0
print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")