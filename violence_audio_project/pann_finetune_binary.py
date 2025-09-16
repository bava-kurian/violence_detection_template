import os
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# ----------------- Config -----------------
DATA_DIR = "dataset"  # Changed from "data" to "dataset" to match split_dataset.py
EPOCHS = 10
BATCH_SIZE = 16
LR = 1e-4
SR = 16000
N_MELS = 64
DURATION = 5
MODEL_PATH = "saved/best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------- Dataset -----------------
class AudioDataset(Dataset):
    def __init__(self, root_dir, sr=SR, n_mels=N_MELS):
        self.samples = []
        self.labels = []
        self.sr = sr
        self.n_mels = n_mels

        for label, folder in enumerate(["non_violence", "violence"]):
            folder_path = os.path.join(root_dir, folder)
            for f in os.listdir(folder_path):
                if f.endswith((".wav", ".mp3", ".mp4")):
                    self.samples.append(os.path.join(folder_path, f))
                    self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]

        # Load audio
        y, sr = librosa.load(file_path, sr=self.sr)
        
        # Make audio fixed length
        target_length = DURATION * self.sr
        if len(y) > target_length:
            # Truncate
            y = y[:target_length]
        else:
            # Pad with zeros
            padding = target_length - len(y)
            y = np.pad(y, (0, padding))

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

        return torch.tensor(log_mel).unsqueeze(0).float(), torch.tensor(label).long()

# ----------------- Model -----------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Calculate the size after conv layers
        mel_length = int(DURATION * SR / 512 + 1)  # 512 is the default hop length in librosa
        conv_output_size = 32 * (N_MELS // 4) * (mel_length // 4)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # binary classification
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----------------- Training -----------------
def train():
    # Create datasets directly from split folders
    train_dataset = AudioDataset(os.path.join(DATA_DIR, "train"))
    val_dataset = AudioDataset(os.path.join(DATA_DIR, "val"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    best_acc = 0
    best_epoch = 0
    train_losses_history = []
    train_acc_history = []
    val_acc_history = []

    print(f"Starting training on {DEVICE}")
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_losses, train_preds, train_labels = [], [], []
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_preds.extend(out.argmax(1).cpu().numpy())
            train_labels.extend(y.cpu().numpy())

            # Print batch progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} - Batch {batch_idx+1}/{len(train_loader)}", end="\r")

        train_loss = np.mean(train_losses)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation phase
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_preds.extend(out.argmax(1).cpu().numpy())
                val_labels.extend(y.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)

        # Update learning rate
        scheduler.step(val_acc)

        # Save histories
        train_losses_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{EPOCHS}:")
        print(f"Loss={train_loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Save model for each epoch
        epoch_path = os.path.join("saved", f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, epoch_path)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, MODEL_PATH)
            print(f"New best model saved! (Validation Accuracy: {val_acc:.3f})")

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

    print(f"\nTraining completed!")
    print(f"Best model was saved at epoch {best_epoch} with validation accuracy: {best_acc:.3f}")

if __name__ == "__main__":
    os.makedirs("saved", exist_ok=True)
    train()
