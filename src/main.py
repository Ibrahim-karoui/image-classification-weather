import os
from ultralytics import YOLO

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
dataset_path = os.path.join(BASE_DIR, "data", "weather_dataset")
model_path = os.path.join(BASE_DIR, "models", "yolo11n-cls.pt")

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

model = YOLO(model_path)
results = model.train(data = dataset_path, epochs = 20 , imgsz = 64)

print("Training finished! Results are saved in 'runs/classify/train'.")