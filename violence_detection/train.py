import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from video_dataset import VideoDataset
from models.i3d_model import get_model
from tqdm import tqdm
import time
import sys  # Added for better exception handling

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
])

train_set = VideoDataset('dataset/train', transform=transform)
val_set = VideoDataset('dataset/val', transform=transform)

train_loader = DataLoader(train_set, batch_size=4, shuffle=True)
val_loader = DataLoader(val_set, batch_size=4)

model = get_model().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

print("Starting training loop...")
try:
    for epoch in range(10):
        print(f"Epoch {epoch+1} starting...")
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            # print(f"Batch loaded")  # Optional: Remove or comment out for less verbosity
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.view(-1)  # Ensures shape [batch_size]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Accuracy calculation (for binary classification)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()

        epoch_time = time.time() - start_time
        accuracy = 100 * correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
        
        # Save the model after each epoch
        torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
except Exception as e:
    print(f"An error occurred during training: {e}", file=sys.stderr)
