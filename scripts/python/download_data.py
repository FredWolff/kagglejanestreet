"""Download competition data from Kaggle to the path defined in config.py."""

import os
import zipfile
from pathlib import Path

from janestreet.config import PATH_DATA
from janestreet import utils

COMPETITION = "jane-street-real-time-market-data-forecasting"


def setup_kaggle_credentials():
    if os.environ.get("KAGGLE_API_TOKEN"):
        print("Kaggle credentials found in environment.")
        return

    print("KAGGLE_API_TOKEN not set. Please enter your Kaggle API token.")
    token = input("Kaggle API token: ").strip()
    os.environ["KAGGLE_API_TOKEN"] = token
    print("Kaggle API token set.")


setup_kaggle_credentials()

utils.create_folder(PATH_DATA)
zip_path = PATH_DATA / f"{COMPETITION}.zip"

print(f"Downloading dataset to {PATH_DATA}...")
os.system(f"kaggle competitions download -c {COMPETITION} -p {PATH_DATA}")

print(f"Unzipping {zip_path}...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(PATH_DATA)

os.remove(zip_path)
print(f"Done. Data available at {PATH_DATA}")
