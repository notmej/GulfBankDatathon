# 🏦 ATM Forecasting Models

## Overview
This project predicts ATM withdrawal activity for Gulf Bank using three baseline forecasting models:

- **Naive Model** — predicts future withdrawals using the most recent value.  
- **Moving Average Model (MA-14)** — predicts using the average of the past 14 days.  
- **Exponential Smoothing Model (SES)** — applies exponentially decreasing weights to historical data.

The workflow is divided into two phases:
1. **Training (`train.py`)** — cleans the dataset and trains all models.
2. **Prediction (`predict.py`)** — generates daily forecasts for each ATM using trained parameters.

---

## How It Works

### 1️. Data Cleaning
Before training, the script `dataCleaning.py`:
- Renames inconsistent columns (e.g. `total_withdrawn_amount_kwd` → `withdrawn_kwd`)
- Drops invalid or duplicate rows  
- Aggregates and aligns daily ATM records  
- Caps outliers at the 99.5th percentile  
- Outputs:  
  - `atm_transactions_train_clean.csv` (cleaned training data)  
  - `features.csv` (optional features for analysis)

### 2️. Model Training
`train.py` runs the entire training pipeline:
- Cleans the raw dataset using `dataCleaning.py`
- Trains:
  - `trainNaive.py` → Naive Model  
  - `trainMovingAvrg.py` → Moving Average Model  
  - `trainExpSmooth.py` → Exponential Smoothing Model  
- Outputs parameter files:
  - `model_naive_params.csv`
  - `modelMovingAvrg_params.csv`
  - `modelExpSmooth_params.csv`

### 3️. Prediction
`predict.py` loads the trained parameters and generates forecasts using:
- `predictNaive.py`
- `predictMovingAvrg.py`
- `predictExpSmooth.py`

Each model outputs its own prediction CSV file:
- `predictions_naive.csv`
- `predictionsMovingAvrg.csv`
- `predictionsExpSmooth.csv`




---

## Output Columns
| Column | Description |
|---------|--------------|
| `dt` | Forecast date |
| `atm_id` | Unique ATM identifier |
| `predicted_withdrawn_kwd` | Forecasted total withdrawn amount (KWD) |
| `predicted_withdraw_count` | Forecasted number of withdrawal transactions |

---

## Model Logic
| Model | Description | Formula |
|--------|--------------|----------|
| **Naive** | Forecast equals last known value | 𝑦̂ₜ₊₁ = 𝑦ₜ |
| **Moving Average (MA-14)** | Forecast is mean of last 14 observations | 𝑦̂ₜ₊₁ = (1/14) Σ 𝑦ₜ₋ᵢ |
| **Exponential Smoothing (SES)** | Uses exponentially weighted mean | 𝑦̂ₜ₊₁ = α𝑦ₜ + (1−α)𝑦̂ₜ |

---

## Installation & Usage

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Train all models
python train.py

### Step 3 — Generate predictions
python predict.py



## Authors:
Icarus:
    Yunona Agzamova
    Boshra Sultan
    Taiba Pattan
    Sakarah Dar 