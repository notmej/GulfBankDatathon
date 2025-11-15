import pandas as pd
df = pd.read_csv("cleaned.csv", parse_dates=["dt"])

# example: constant forecast using MA(14) per ATM for withdrawn & count
def ma14(g, col):
    return g[col].tail(14).mean()

last_dt = df["dt"].max()
future = pd.DataFrame({"dt": pd.date_range(last_dt + pd.Timedelta(days=1), periods=14, freq="D")})
rows = []
for atm, g in df.groupby("atm_id"):
    g = g.sort_values("dt")
    kwd = ma14(g, "withdrawn_kwd")
    cnt = ma14(g, "withdraw_count")
    tmp = future.copy()
    tmp["atm_id"] = atm
    tmp["predicted_withdrawn_kwd"] = max(0.0, kwd) if pd.notna(kwd) else 0.0
    tmp["predicted_withdraw_count"] = max(0.0, cnt) if pd.notna(cnt) else 0.0
    rows.append(tmp)

out = pd.concat(rows, ignore_index=True)
out = out[["dt","atm_id","predicted_withdrawn_kwd","predicted_withdraw_count"]]
out.to_csv("predictions.csv", index=False)
print("wrote predictions.csv", out.shape)
