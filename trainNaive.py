import pandas as pd

def main():
    # Load the training data
    df = pd.read_csv("atm_transactions_train.csv", parse_dates=["dt"])

    # Expected columns
    required = ["dt", "atm_id", "total_withdrawn_amount_kwd", "total_withdraw_txn_count"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    # Clean and prep
    df = df.dropna(subset=["atm_id", "dt"]).sort_values(["atm_id", "dt"])
    df["total_withdrawn_amount_kwd"] = pd.to_numeric(df["total_withdrawn_amount_kwd"], errors="coerce").fillna(0).clip(lower=0)
    df["total_withdraw_txn_count"] = pd.to_numeric(df["total_withdraw_txn_count"], errors="coerce").fillna(0).clip(lower=0)

    # --- Naive model training ---
    model_rows = []
    for atm, g in df.groupby("atm_id"):
        g = g.sort_values("dt")
        last_kwd = g["total_withdrawn_amount_kwd"].iloc[-1]
        last_cnt = g["total_withdraw_txn_count"].iloc[-1]
        model_rows.append({
            "atm_id": atm,
            "last_withdrawn_kwd": last_kwd,
            "last_withdraw_count": last_cnt
        })

    model = pd.DataFrame(model_rows).sort_values("atm_id").reset_index(drop=True)
    model.to_csv("model_naive_params.csv", index=False)

    print(f"✅ Naive model trained successfully — saved {len(model)} ATMs to model_naive_params.csv")
    print(model.head())

if __name__ == "__main__":
    main()
