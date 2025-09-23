import os
import numpy as np
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import cv2

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
test_dir = os.path.join(BASE_DIR, "data", "weather_dataset", "test")
model_path = os.path.join(BASE_DIR, "runs", "classify", "train", "weights", "best.pt")

model = YOLO(model_path)

y_true = []
y_pred = []

# dict label -> id
class_to_idx = {v: k for k, v in model.names.items()}

# Go through all the images in the test folder
for root, dirs, files in os.walk(test_dir):
    label = os.path.basename(root)
    if label not in class_to_idx:
        continue

    true_class = class_to_idx[label] 

    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, file)

            # verify if the image exists
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            results = model(img_path)

            pred_class = results[0].probs.top1  # predicted class

            y_true.append(true_class)
            y_pred.append(pred_class)

# Calculate accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)
