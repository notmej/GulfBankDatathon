import pandas as pd
import numpy as np

# -----------------------------
# 1) Load data
# -----------------------------
df = pd.read_csv("cleaned.csv", parse_dates=["dt"])
pred = pd.read_csv("predictions.csv", parse_dates=["dt"])
print(f"Loaded actuals: {df.shape}, predictions: {pred.shape}")

# Required columns
need_actuals = ["dt", "atm_id", "withdrawn_kwd", "withdraw_count"]
need_pred = ["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]

miss_a = [c for c in need_actuals if c not in df.columns]
miss_p = [c for c in need_pred if c not in pred.columns]
if miss_a:
    raise ValueError(f"cleaned.csv missing columns: {miss_a}")
if miss_p:
    raise ValueError(f"predictions.csv missing columns: {miss_p}")

df = df[need_actuals].copy()
pred = pred[need_pred].copy()

# -----------------------------
# 2) Clean actuals
# -----------------------------
df = df.dropna(subset=["atm_id", "dt"])
df["withdrawn_kwd"] = pd.to_numeric(df["withdrawn_kwd"], errors="coerce")
df["withdraw_count"] = pd.to_numeric(df["withdraw_count"], errors="coerce")

df = df.sort_values(["atm_id", "dt"]).drop_duplicates(["atm_id", "dt"], keep="last")

df[["withdrawn_kwd", "withdraw_count"]] = (
    df.groupby("atm_id", group_keys=False)[["withdrawn_kwd", "withdraw_count"]]
      .apply(lambda g: g.ffill().bfill())
      .fillna(0)
)
df["withdrawn_kwd"] = df["withdrawn_kwd"].clip(lower=0)
df["withdraw_count"] = df["withdraw_count"].clip(lower=0)

# -----------------------------
# 3) Clean predictions
# -----------------------------
pred = pred.dropna(subset=["atm_id", "dt"])
pred["predicted_withdrawn_kwd"] = pd.to_numeric(pred["predicted_withdrawn_kwd"], errors="coerce")
pred["predicted_withdraw_count"] = pd.to_numeric(pred["predicted_withdraw_count"], errors="coerce")

pred = pred.sort_values(["atm_id", "dt"]).drop_duplicates(["atm_id", "dt"], keep="last")

# Ensure complete ATM × date grid
atm_ids = np.sort(df["atm_id"].unique())
forecast_dates = np.sort(pred["dt"].unique())

expected = len(atm_ids) * len(forecast_dates)
missing_rows = expected - len(pred)

if missing_rows > 0:
    print(f"[Predictions] filling {missing_rows} missing (atm_id, dt) pairs with 0s")
    full_index = pd.MultiIndex.from_product([atm_ids, forecast_dates], names=["atm_id", "dt"])
    pred = pred.set_index(["atm_id", "dt"]).reindex(full_index).reset_index()

pred["predicted_withdrawn_kwd"] = pred["predicted_withdrawn_kwd"].fillna(0).clip(lower=0)
pred["predicted_withdraw_count"] = pred["predicted_withdraw_count"].fillna(0).clip(lower=0).round().astype(int)

pred = pred.sort_values(["atm_id", "dt"]).reset_index(drop=True)

# -----------------------------
# 4) Align actuals & predictions
# -----------------------------
merged = pd.merge(df, pred, on=["atm_id", "dt"], how="inner", validate="1:1").sort_values(["atm_id", "dt"])

# -----------------------------
# 5) RMSE (NumPy)
# -----------------------------
def rmse_np(actual: pd.Series, pred: pd.Series) -> float:
    a = actual.to_numpy(dtype=float)
    p = pred.to_numpy(dtype=float)
    mask = ~np.isnan(a) & ~np.isnan(p)
    if mask.sum() == 0:
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - p[mask]) ** 2)))

rmse_kwd = rmse_np(merged["withdrawn_kwd"], merged["predicted_withdrawn_kwd"])
rmse_cnt = rmse_np(merged["withdraw_count"], merged["predicted_withdraw_count"])

print("\n📊 RMSE RESULTS")
print(f" - Withdrawn KWD:  {rmse_kwd:.6f}")
print(f" - Withdraw Count: {rmse_cnt:.6f}")

# -----------------------------
# 6) Sanity checks
# -----------------------------
print("\n🔍 Final Checks")
print(f"Duplicates in merged: {merged.duplicated(subset=['atm_id','dt']).sum()}")
print("Missing values:")
print(merged[["withdrawn_kwd","withdraw_count","predicted_withdrawn_kwd","predicted_withdraw_count"]].isna().sum())

print("\n✅ Evaluation complete (no CSVs written).")
