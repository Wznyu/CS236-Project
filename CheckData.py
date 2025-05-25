import pandas as pd
import json
import os

data_dir = "Datasets"
files = ["AmazonElectronicsMetadata.csv", "Electronics_5.json", "DatafinitiElectronicsProductData.csv"]

for file in files:
    path = os.path.join(data_dir, file)
    print(f"File: {file}")
    
    try:
        if file.endswith(".csv"):
            df = pd.read_csv(path)
        elif file.endswith(".json"):
            df = pd.read_json(path, lines=True)

        columns = df.columns
        print(f"  Number of columns: {len(columns)}")
        print(f"  Column names: {list(columns)}\n")

    except Exception as e:
        print(f"  Failed to load {file}: {e}\n")

