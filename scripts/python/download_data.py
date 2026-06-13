"""Download competition data from Kaggle to the path defined in config.py."""

import json
import os
import zipfile
from pathlib import Path

from janestreet.config import PATH_DATA
from janestreet import utils

COMPETITION = "jane-street-real-time-market-data-forecasting"
KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_CREDENTIALS = KAGGLE_DIR / "kaggle.json"


def setup_kaggle_credentials():
    if KAGGLE_CREDENTIALS.exists():
        print("Kaggle credentials already configured.")
        return

    print("Kaggle credentials not found. Please enter them now.")
    username = input("Kaggle username: ").strip()
    token = input("Kaggle API token: ").strip()

    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(KAGGLE_CREDENTIALS, "w") as f:
        json.dump({"username": username, "key": token}, f)
    KAGGLE_CREDENTIALS.chmod(0o600)
    print(f"Credentials saved to {KAGGLE_CREDENTIALS}")


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
