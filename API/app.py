from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from safetensors.torch import load_file
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

app = Flask(__name__)
CORS(app)

# Load your trained model once at startup
MODEL_PATH = '.'  # directory containing model.safetensors, config.json, preprocessor_config.json

model = VideoMAEForVideoClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
processor = VideoMAEImageProcessor.from_pretrained(MODEL_PATH)
model.eval()

def preprocess_frames(frames):
    # Example preprocessing: resize, convert to tensor, normalize
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    processed = [preprocess(frame[:, :, ::-1]) for frame in frames]  # Convert BGR to RGB
    input_tensor = torch.stack(processed)  # Shape: (num_frames, 3, 224, 224)
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension: (1, num_frames, 3, 224, 224)
    return input_tensor

def predict_violence(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return "error: no frames"
    # Convert frames to RGB
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    # Use processor to prepare input
    inputs = processor(frames_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_label = outputs.logits.argmax(-1).item()
    # Map label to class name if available
    id2label = model.config.id2label if hasattr(model.config, "id2label") else {0: "non-violent", 1: "violent"}
    prediction = id2label.get(predicted_label, str(predicted_label))
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a POST request to /predict")  # Add this
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    os.makedirs('uploads', exist_ok=True)
    file.save(filepath)
    prediction = predict_violence(filepath)
    os.remove(filepath)
    print(f"Prediction for {filename}: {prediction}")  # <-- This will print to your terminal
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)