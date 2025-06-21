from transformers import pipeline

# Load the model (after installing PyAV)
classifier = pipeline("video-classification", model="Nikeytas/videomae-crime-violence-detector")

# Test on a video file
result = classifier("V_16.mp4")  # Replace with your video
print("Prediction:", result[0]["label"], "| Score:", result[0]["score"])