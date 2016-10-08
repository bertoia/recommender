"""
Reduce the decimal place of model prediction to save space
"""
import os
import pandas as pd
decimal = 4
directory = r"data\predictions"
files = [os.path.join(directory, file) for file in os.listdir(directory)
         if file.endswith(".csv")]

for file in files:
    print(file)
    data = pd.read_csv(file)
    data.round(4).to_csv(file, index=False)
