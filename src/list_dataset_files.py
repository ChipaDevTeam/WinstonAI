import os

dataset_path = r"C:\Users\vigop\.cache\kagglehub\datasets\komodata\forexdataset\versions\2"

if os.path.exists(dataset_path):
    print(f"Listing files in {dataset_path}:")
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            print(os.path.join(root, file))
else:
    print(f"Path does not exist: {dataset_path}")
