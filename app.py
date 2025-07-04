import streamlit as st
import torch
from torchvision import transforms
from violence_detection.video_dataset import load_video
from violence_detection.models.i3d_model import get_model
import tempfile
import os
import numpy as np
import cv2

st.title("Violence Detection in Video")
st.write("Upload a video file to detect violence.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"]) 

if uploaded_file is not None:
    # Save uploaded file to a temp location
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()  # <-- Close the file handle!
    video_path = tfile.name

    # Show video
    st.video(video_path)

    # Model and device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.join("violence_detection", "model_epoch_10.pth")
    model = get_model().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    # Preprocess video
    try:
        frames = load_video(video_path)  # returns list of tensors (C, H, W)
        frames = torch.stack([transform(frame) for frame in frames])  # (T, C, H, W)
        video_tensor = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # (1, C, T, H, W)

        with torch.no_grad():
            output = model(video_tensor)
            output = output.squeeze().item()
            confidence = float(output)
            prediction = 1 if confidence > 0.5 else 0

        st.markdown(f"**Prediction:** {'Violence' if prediction == 1 else 'Non-violence'}")
        st.markdown(f"**Confidence:** {confidence:.4f}")
    except Exception as e:
        st.error(f"Error processing video: {e}")
    finally:
        # os.unlink(video_path)  # <-- Comment out or remove this line
        pass 