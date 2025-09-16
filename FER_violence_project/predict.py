import cv2
import torch
from torchvision import transforms, models
from PIL import Image
import numpy as np
import argparse
import os

# Set up argument parser
parser = argparse.ArgumentParser(description="Emotion recognition from image or webcam")
parser.add_argument("--image", type=str, default=None,
                    help="Path to input image (if not provided, webcam will be used).")
args = parser.parse_args()

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Emotion classes
class_names = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load model
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define transform
transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def predict_emotion(face_img):
    face_img = Image.fromarray(face_img)
    face_img = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(face_img)
        _, predicted = torch.max(output, 1)
    return class_names[predicted.item()]

def run_on_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}")
        return
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        face = img_rgb[y:y+h, x:x+w]
        emotion = predict_emotion(face)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img_bgr, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    cv2.imshow("Emotion Recognition - Image", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def run_webcam():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            emotion = predict_emotion(face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow('Emotion Recognition - Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if args.image:
        run_on_image(args.image)
    else:
        run_webcam()
