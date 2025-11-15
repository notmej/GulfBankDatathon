import pandas as pd

def main():
    # Load test and trained model
    test = pd.read_csv("atm_transactions_test.csv", parse_dates=["dt"])
    model = pd.read_csv("model_naive_params.csv")

    # Validate structure
    required = ["dt", "atm_id"]
    missing = [c for c in required if c not in test.columns]
    if missing:
        raise ValueError(f"Test data missing columns: {missing}")

    # Prepare test data
    test = test.dropna(subset=["atm_id", "dt"]).sort_values(["atm_id", "dt"]).reset_index(drop=True)

    # Merge last known values for each ATM
    pred = pd.merge(test, model, on="atm_id", how="left")

    # Fill missing model values (for unseen ATMs)
    pred["last_withdrawn_kwd"] = pred["last_withdrawn_kwd"].fillna(0)
    pred["last_withdraw_count"] = pred["last_withdraw_count"].fillna(0)

    # Create predicted columns
    pred["predicted_withdrawn_kwd"] = pred["last_withdrawn_kwd"].clip(lower=0)
    pred["predicted_withdraw_count"] = pred["last_withdraw_count"].clip(lower=0).round().astype(int)

    # Output final dataframe
    out = pred[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]]
    out.to_csv("predictions_naive.csv", index=False)

    print(f"✅ wrote predictions_naive.csv, shape={out.shape}")
    print(out.head())

if __name__ == "__main__":
    main()
