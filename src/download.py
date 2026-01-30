import kagglehub

# Download latest version
path = kagglehub.dataset_download("komodata/forexdataset")

print("Path to dataset files:", path)