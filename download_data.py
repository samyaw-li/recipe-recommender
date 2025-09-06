import os
from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "shuyangli94/food-com-recipes-and-user-interactions"
DATA_DIR = "data"

def download_data():
    # make sure "data/" folder exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # connect to Kaggle API
    api = KaggleApi()
    api.authenticate()

    # download + unzip dataset
    print(f"Downloading dataset {DATASET}...")
    api.dataset_download_files(DATASET, path=DATA_DIR, unzip=True)
    print("✅ Download complete! Files are in the 'data/' folder.")

if __name__ == "__main__":
    download_data()
