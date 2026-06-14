# Vast.ai Setup Guide

## 1. Rent an Instance

- Go to vast.ai and find a machine with a GPU
- Set the Docker image to: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
- Make sure the CUDA version in the image matches the host driver version shown on the listing
- Start the instance and SSH in

## 2. Clone the Repository

```bash
git clone https://GITHUB_TOKEN@github.com/FredWolff/kagglejanestreet.git /home/janestreet2024
cd /home/janestreet2024
```

## 3. Install Dependencies

```bash
pip install -e .
```

## 4. Set Up Environment Variables

```bash
echo "WANDB_TOKEN=your_wandb_token" > .env
```

Your wandb token is available at: `wandb.ai → Settings → API keys`

## 5. Download the Data

```bash
python scripts/python/download_data.py
```

You will be prompted for your Kaggle credentials:
- **Username**: your Kaggle username
- **API token**: found at `kaggle.com → Settings → API → Create New Token`

Note: You must have accepted the competition rules at
`kaggle.com/competitions/jane-street-real-time-market-data-forecasting`
before the download will work.

## 6. Run Training

Cross-validation:
```bash
python scripts/python/run_cv.py
```

Full training (model 2):
```bash
python scripts/python/run_full_2.py
```

Full training (model 3):
```bash
python scripts/python/run_full_3.py
```

Ensemble evaluation:
```bash
python scripts/python/run_ensemble.py
```
