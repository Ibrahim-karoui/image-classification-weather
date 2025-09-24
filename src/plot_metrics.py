import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
results_path = os.path.join(BASE_DIR, "runs", "classify", "train", "results.csv")

results = pd.read_csv(results_path, sep=",")

print(results.columns.tolist())

# Plot train/val loss
plt.figure()
plt.plot(results['epoch'], results['train/loss'], label='train loss')
plt.plot(results['epoch'], results['val/loss'], label='val loss', c='red')
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

# Plot accuracy top1
plt.figure()
plt.plot(results['epoch'], results['metrics/accuracy_top1'] * 100)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy vs Epochs')

plt.show()
