"""Download competition data from Kaggle to the path defined in config.py."""

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from janestreet.config import PATH_DATA
from janestreet import utils

COMPETITION = "jane-street-real-time-market-data-forecasting"


def get_kaggle_executable() -> str:
    """Locate the kaggle CLI, preferring the one installed alongside this interpreter.

    os.system() spawns a fresh shell that may not have a venv's Scripts/bin
    directory on PATH, so plain "kaggle" can fail to resolve even when it's
    installed. Falls back to a plain PATH lookup to preserve existing
    behavior when kaggle is installed globally (e.g. on Kaggle/vast.ai).
    """
    exe_name = "kaggle.exe" if os.name == "nt" else "kaggle"
    candidate = Path(sys.executable).parent / exe_name
    if candidate.exists():
        return str(candidate)
    return shutil.which("kaggle") or "kaggle"


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
kaggle_exe = get_kaggle_executable()
subprocess.run(
    [kaggle_exe, "competitions", "download", "-c", COMPETITION, "-p", str(PATH_DATA)],
    check=True,
)

print(f"Unzipping {zip_path}...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(PATH_DATA)

os.remove(zip_path)
print(f"Done. Data available at {PATH_DATA}")
