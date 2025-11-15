import pandas as pd
import numpy as np

def main():
    PARAMS_FILE = "modelExpSmooth_params.csv"

    # Load model and test data
    params = pd.read_csv(PARAMS_FILE, parse_dates=["last_train_dt"])
    test = pd.read_csv("atm_transactions_test.csv", parse_dates=["dt"])

    # Validate structure
    required_test = ["dt", "atm_id"]
    missing = [c for c in required_test if c not in test.columns]
    if missing:
        raise ValueError(f"Test data missing columns: {missing}")

    # Clean and prep
    test = (
        test[required_test]
        .dropna(subset=["atm_id", "dt"])
        .drop_duplicates(["atm_id", "dt"])
        .sort_values(["atm_id", "dt"])
        .reset_index(drop=True)
    )

    # Merge SES levels
    pred = test.merge(params[["atm_id", "ses_level_kwd", "ses_level_cnt"]], on="atm_id", how="left")
    pred["ses_level_kwd"] = pd.to_numeric(pred["ses_level_kwd"], errors="coerce").fillna(0).clip(lower=0)
    pred["ses_level_cnt"] = pd.to_numeric(pred["ses_level_cnt"], errors="coerce").fillna(0).clip(lower=0)

    # Forecasts (constant SES level)
    pred["predicted_withdrawn_kwd"] = pred["ses_level_kwd"]
    pred["predicted_withdraw_count"] = pred["ses_level_cnt"].round().astype(int)

    # Save predictions
    out = pred[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]].sort_values(["atm_id", "dt"]).reset_index(drop=True)
    out.to_csv("predictionsExpSmooth.csv", index=False)

    print(f"✅ Wrote predictionsExpSmooth.csv, shape={out.shape}")
    print(out.head())

if __name__ == "__main__":
    main()
