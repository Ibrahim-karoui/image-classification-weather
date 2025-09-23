import os
from ultralytics import YOLO
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

model_path = os.path.join(BASE_DIR, "runs", "classify", "train", "weights", "best.pt")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = YOLO(model_path)

# put the name of the image to test and do not forget to change the name of the class
test_image = os.path.join(BASE_DIR, "data", "weather_dataset", "test", "sunrise", "image4.jpg")

if not os.path.exists(test_image):
    raise FileNotFoundError(f"Test image not found at {test_image}")

# Prediction
results = model(test_image)

names_dict = results[0].names
probs = results[0].probs.data.cpu().numpy().tolist() 

print("*****************************************************\n")
print(f"Image: {os.path.basename(test_image)}")
print("Classes:", names_dict)
print("Probabilities:", probs)
print("Top-1 class:", names_dict[results[0].probs.top1], "=>", float(results[0].probs.top1conf))
print("*****************************************************\n")
