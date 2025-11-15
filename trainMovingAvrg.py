import pandas as pd
import numpy as np

def main():
    WINDOW = 14  # number of days for moving average

    # Load and keep only needed columns
    df = pd.read_csv("atm_transactions_train.csv", parse_dates=["dt"])
    df = df[["dt", "atm_id", "total_withdrawn_amount_kwd", "total_withdraw_txn_count"]].copy()

    # Clean data
    df = df.dropna(subset=["atm_id", "dt"])
    df["total_withdrawn_amount_kwd"] = pd.to_numeric(df["total_withdrawn_amount_kwd"], errors="coerce").fillna(0).clip(lower=0)
    df["total_withdraw_txn_count"] = pd.to_numeric(df["total_withdraw_txn_count"], errors="coerce").fillna(0).clip(lower=0)
    df = df.sort_values(["atm_id", "dt"]).drop_duplicates(["atm_id", "dt"], keep="last")

    # --- Learn moving average per ATM ---
    def learn_ma(group, window):
        recent = group.tail(window)
        avg_kwd = float(np.nanmean(recent["total_withdrawn_amount_kwd"]))
        avg_cnt = float(np.nanmean(recent["total_withdraw_txn_count"]))
        return {
            "atm_id": group["atm_id"].iloc[0],
            "ma_kwd": max(0.0, avg_kwd if np.isfinite(avg_kwd) else 0.0),
            "ma_cnt": max(0.0, avg_cnt if np.isfinite(avg_cnt) else 0.0),
            "window": window,
            "n_used": len(recent)
        }

    params = [learn_ma(g, WINDOW) for _, g in df.groupby("atm_id")]
    model = pd.DataFrame(params).sort_values("atm_id").reset_index(drop=True)
    model.to_csv("modelMovingAvrg_params.csv", index=False)

    print(f"✅ Trained Moving Average model ({WINDOW}-day window) for {len(model)} ATMs")
    print(model.head())

if __name__ == "__main__":
    main()
