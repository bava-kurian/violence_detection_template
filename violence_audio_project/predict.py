import sys
import subprocess
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pann_finetune_binary import SimpleCNN, SR, N_MELS, MODEL_PATH, DEVICE, DURATION

def extract_audio_ffmpeg(video_path, out_path="temp_audio.wav"):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",             # drop video
        "-acodec", "pcm_s16le",
        "-ar", str(SR),    # sample rate
        "-ac", "1",        # mono
        out_path
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def predict(file_path):
    # If it's a video, first extract audio
    if file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        file_path = extract_audio_ffmpeg(file_path)

    # Load audio
    y, sr = librosa.load(file_path, sr=SR)
    
    # Make audio fixed length like in training
    target_length = DURATION * SR
    if len(y) > target_length:
        y = y[:target_length]
    else:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding))
    
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)

    # Plot spectrogram
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(log_mel, sr=sr, x_axis="time", y_axis="mel")
    plt.title("Spectrogram")
    plt.colorbar(format="%+2.f dB")
    plt.savefig("spectrogram.png")   # saves the figure instead of blocking
    plt.close()


    # Prepare input
    x = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    # Load model with modified loading parameters
    model = SimpleCNN().to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])  # Extract just the model weights
    model.eval()

    with torch.no_grad():
        out = model(x)
        pred = out.argmax(1).item()
        prob = torch.softmax(out, dim=1)[0][pred].item()

    label = "Violent" if pred == 1 else "Non-Violent"
    return label, prob

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio_or_video>")
        sys.exit(1)

    file_path = sys.argv[1]
    label, prob = predict(file_path)
    print(f"Prediction: {label} (confidence: {prob:.2f})")
