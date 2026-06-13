"""Configuration file."""

import os
from pathlib import Path

# Determine the execution environment based on environment variables
KAGGLE = 'KAGGLE_URL_BASE' in os.environ
VASTAI = not KAGGLE

# Define base paths for different environments (change if needed)
# Data and models can be stored on a different volume
# Path with data should contain subdirectoriy "data" 
# with data from https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting/data
base_paths = {
    "VASTAI": Path("/home/janestreet2024"),
    "VASTAI_DATA": Path("/workspace/kaggle/janestreet"),
    "KAGGLE": Path("/kaggle/input"),
}

# Set paths based on the environment (change if needed)
if VASTAI:
    base_path = base_paths["VASTAI"]
    base_path_data = base_paths["VASTAI_DATA"]
    PATH_DATA = base_path_data / "data"
    PATH_MODELS = base_path_data / "models"
    PATH_CODE = base_path / "dist/janestreet-0.1-py3-none-any.whl"
elif KAGGLE:
    base_path = base_paths["KAGGLE"]
    base_path_data = base_paths["KAGGLE"]
    PATH_DATA = base_path / "jane-street-real-time-market-data-forecasting"
    PATH_MODELS = base_path / "janestreet2025-models"
    PATH_CODE = base_path / "janestreet2025-code/janestreet-0.1-py3-none-any.whl"
else:
    raise ValueError("Unknown environment")

PATHS_DATA = {
    "train": PATH_DATA / "train",
    "test": PATH_DATA / "test",
}

# Set other configuration variables
# Wandb (to track experiments, not required)
WANDB_PROJECT = "kaggle_janestreet"

# Kaggle (to push code and models, not required)
KAGGLE_USERNAME = "frederikwolff"

# Default random seed
RANDOM_SEED = 42

# Data column names (do not change)
COL_TARGET = "responder_6"
COL_ID = "symbol_id"
COL_DATE = "date_id"
COL_TIME = "time_id"
COL_WEIGHT = "weight"
COLS_RESPONDERS = [f"responder_{i}" for i in range(11)]
