import pandas as pd
import os

file_path = r"C:\Users\vigop\.cache\kagglehub\datasets\komodata\forexdataset\versions\2\EURUSD_M30.csv"

if os.path.exists(file_path):
    try:
        df = pd.read_csv(file_path)
        print("Columns:", df.columns.tolist())
        print(df.head())
    except Exception as e:
        print(f"Error reading csv: {e}")
else:
    print("File not found.")
