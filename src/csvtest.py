import pandas as pd

results_path = r"C:\Users\brahi\Documents\Computer_vision\Image classification weather\runs\classify\train\results.csv"
results = pd.read_csv(results_path, sep=",", skiprows=1)
print(results.columns.tolist())
print(results.head())