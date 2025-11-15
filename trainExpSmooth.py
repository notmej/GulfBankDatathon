import pandas as pd
import numpy as np

def main():
    ALPHAS = np.linspace(0.05, 0.95, 19)
    REQUIRED = ["dt", "atm_id", "total_withdrawn_amount_kwd", "total_withdraw_txn_count"]

    # Load & clean data
    df = pd.read_csv("atm_transactions_train.csv", parse_dates=["dt"])
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in training data: {missing}")

    df = df[REQUIRED].dropna(subset=["atm_id", "dt"]).sort_values(["atm_id", "dt"])
    df["total_withdrawn_amount_kwd"] = pd.to_numeric(df["total_withdrawn_amount_kwd"], errors="coerce").fillna(0).clip(lower=0)
    df["total_withdraw_txn_count"] = pd.to_numeric(df["total_withdraw_txn_count"], errors="coerce").fillna(0).clip(lower=0)

    # --- Simple Exponential Smoothing ---
    def fit_ses(y, alphas=ALPHAS):
        y = y.astype(float)
        yy = y[~np.isnan(y)]
        n = len(yy)
        if n == 0:
            return 0.0, 0.3, 0
        if n == 1:
            return float(max(0.0, yy[0])), 0.3, 1

        best_alpha, best_sse = None, np.inf
        for a in alphas:
            level = yy[0]
            sse = 0.0
            for t in range(1, n):
                y_t = yy[t]
                err = y_t - level
                sse += err * err
                level = a * y_t + (1 - a) * level
            if sse < best_sse:
                best_sse, best_alpha = sse, a

        # final level
        level = yy[0]
        for t in range(1, n):
            level = best_alpha * yy[t] + (1 - best_alpha) * level
        return float(max(0.0, level)), float(best_alpha), int(n)

    rows = []
    for atm, g in df.groupby("atm_id", sort=True):
        g = g.sort_values("dt")
        amt = g["total_withdrawn_amount_kwd"].to_numpy()
        cnt = g["total_withdraw_txn_count"].to_numpy()

        level_amt, alpha_amt, n_amt = fit_ses(amt)
        level_cnt, alpha_cnt, n_cnt = fit_ses(cnt)

        rows.append({
            "atm_id": atm,
            "ses_level_kwd": level_amt,
            "ses_level_cnt": level_cnt,
            "alpha_kwd": alpha_amt,
            "alpha_cnt": alpha_cnt,
            "n_used_kwd": n_amt,
            "n_used_cnt": n_cnt,
            "last_train_dt": g["dt"].max()
        })

    model = pd.DataFrame(rows).sort_values("atm_id").reset_index(drop=True)
    model.to_csv("modelExpSmooth_params.csv", index=False)

    print(f"✅ Trained SES model for {len(model)} ATMs → modelExpSmooth_params.csv")
    print(model.head())

if __name__ == "__main__":
    main()
