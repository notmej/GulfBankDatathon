import pandas as pd
import numpy as np

def main():
    PARAMS_FILE = "modelMovingAvrg_params.csv"

    # Load model and test data
    model = pd.read_csv(PARAMS_FILE)
    test = pd.read_csv("atm_transactions_test.csv", parse_dates=["dt"])

    # Keep only needed columns
    test = test[["dt", "atm_id"]].dropna(subset=["atm_id", "dt"]).drop_duplicates(["atm_id", "dt"])
    test = test.sort_values(["atm_id", "dt"]).reset_index(drop=True)

    # Merge learned parameters
    pred = pd.merge(test, model, on="atm_id", how="left")

    # Handle missing (unseen ATMs)
    pred["ma_kwd"] = pd.to_numeric(pred["ma_kwd"], errors="coerce").fillna(0)
    pred["ma_cnt"] = pd.to_numeric(pred["ma_cnt"], errors="coerce").fillna(0)

    # Create predictions
    pred["predicted_withdrawn_kwd"] = pred["ma_kwd"].clip(lower=0)
    pred["predicted_withdraw_count"] = pred["ma_cnt"].clip(lower=0).round().astype(int)

    # Output clean file
    out = pred[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]]
    out = out.sort_values(["atm_id", "dt"]).reset_index(drop=True)
    out.to_csv("predictionsMovingAvrg.csv", index=False)

    print(f"✅ Predictions generated: {out.shape[0]} rows, saved to predictionsMovingAvrg.csv")
    print(out.head())

if __name__ == "__main__":
    main()
