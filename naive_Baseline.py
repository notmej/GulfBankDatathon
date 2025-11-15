import pandas as pd

# --- Load your cleaned dataset ---
df = pd.read_csv("cleaned.csv", parse_dates=["dt"])

# --- Get the last available date ---
last_dt = df["dt"].max()
print("Last available date:", last_dt)

# --- Create a 14-day forecast horizon ---
future_dates = pd.date_range(last_dt + pd.Timedelta(days=1), periods=14, freq="D")

# --- Prepare an empty list to collect forecasts ---
rows = []

for atm, g in df.groupby("atm_id"):
    g = g.sort_values("dt")

    # last known values (Naive model)
    last_kwd = g["withdrawn_kwd"].iloc[-1]
    last_cnt = g["withdraw_count"].iloc[-1]

    tmp = pd.DataFrame({
        "dt": future_dates,
        "atm_id": atm,
        "predicted_withdrawn_kwd": [last_kwd] * len(future_dates),
        "predicted_withdraw_count": [last_cnt] * len(future_dates)
    })
    rows.append(tmp)

# --- Combine and save ---
out = pd.concat(rows, ignore_index=True)
out = out[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]]
out.to_csv("predictions_naive.csv", index=False)

print("✅ predictions_naive.csv created successfully")
print(out.head(10))
