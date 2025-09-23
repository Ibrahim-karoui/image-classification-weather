# Image Classification Weather

This project implements an image classification system for different weather conditions using **YOLOv11 classification model** with **Ultralytics**, **PyTorch**, and **Python**. The model can classify images into four weather classes: **rain, sunrise, shine, and cloudy**.

## Project Structure

Image Classification Weather/
│
├─ data/
│  └─ weather_dataset/
│     ├─ train/
│     │  ├─ rain/
│     │  ├─ sunrise/
│     │  ├─ shine/
│     │  └─ cloudy/
│     ├─ val/
│     │  ├─ rain/
│     │  ├─ sunrise/
│     │  ├─ shine/
│     │  └─ cloudy/
│     └─ test/
│        ├─ rain/
│        ├─ sunrise/
│        ├─ shine/
│        └─ cloudy/
│
├─ models/
│  └─ yolo11n-cls.pt
│
├─ runs/
│  └─ classify/           # Training results will be saved here
│
├─ src/
│  ├─ main.py             # Training script
│  ├─ predict.py          # Predict a single image
│  ├─ evaluate.py         # Evaluate model on the test dataset
│  └─ plot_metrics.py     # Plot training loss and accuracy
│
└─ requirements.txt       # Python dependencies

## Installation

### Clone the repository:
git clone https://github.com/yourusername/Image-Classification-Weather.git
cd Image-Classification-Weather

### Create and activate a virtual environment:
python -m venv env
# Windows
.\env\Scripts\activate
# macOS/Linux
source env/bin/activate

### Install dependencies:
pip install -r requirements.txt

## Usage
### Training the model
To train the YOLOv11 classification model:
python src/main.py

Training results, including weights and logs, will be saved under runs/classify/train/.
The dataset should be organized as:

data/weather_dataset/
├─ train/
├─ val/

Place the image in the data/weather_dataset/test/<class_name>/ folder or update the path in predict.py.
The script prints the predicted class and probabilities.

### Evaluating the model
To evaluate the model on the test dataset:

python src/evaluate.py

The test dataset must be organized with subfolders for each class.
The script calculates accuracy and prints the confusion matrix.

### Dataset
The dataset consists of images of four weather conditions:
rain
sunrise
shine
cloudy

The dataset is split into train, val, and test folders, each containing subfolders for each class.

Dataset link : https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4drtyfjtfy-1.zip

### Dependencies
Python 3.12+
PyTorch
Ultralytics YOLO
OpenCV
Matplotlib
Pandas
scikit-learn

(See requirements.txt for exact versions.)

## Acknowledgements
This project was developed by following the tutorial from the YouTube channel "Computer Vision Engineer" with some modifications
channel link = "https://www.youtube.com/@ComputerVisionEngineer"


