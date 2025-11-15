# clean_atm_data.py  —  robust & Windows-friendly
# Run:
#   python clean_atm_data.py
# or:
#   python clean_atm_data.py --input "atm_transactions_train.csv" --out-clean "cleaned.csv" --out-features "features.csv" --weekend "4,5"

import argparse
import sys
import os
import pandas as pd
import numpy as np

DEFAULT_INPUT = "atm_transactions_train.csv"
DEFAULT_OUT_CLEAN = "cleaned.csv"
DEFAULT_OUT_FEATURES = "features.csv"
DEFAULT_WEEKEND = {4, 5}  # Fri(4), Sat(5) for Kuwait; Monday=0

RENAME_MAP = {
    "total_withdrawn_amount_kwd": "withdrawn_kwd",
    "total_withdraw_txn_count": "withdraw_count",
    "total_deposited_amount_kwd": "deposited_kwd",
    "total_deposit_txn_count": "deposit_count",
}
NUMERIC_COLS = ["withdrawn_kwd", "withdraw_count", "deposited_kwd", "deposit_count"]

def eprint(*a):
    print(*a, file=sys.stderr)

def parse_weekend(arg: str):
    if not arg:
        return DEFAULT_WEEKEND
    try:
        vals = {int(x.strip()) for x in arg.split(",") if x.strip() != ""}
        # validate 0..6
        vals = {v for v in vals if 0 <= v <= 6}
        return vals or DEFAULT_WEEKEND
    except Exception:
        eprint("⚠️  Could not parse --weekend. Falling back to default (4,5).")
        return DEFAULT_WEEKEND

def read_and_standardize(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    # Try utf-8-sig first (handles BOM); fallback to utf-8
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        df = pd.read_csv(path, encoding="utf-8")

    df.columns = [c.lower().strip() for c in df.columns]

    # Show what we got
    print("🔎 Columns found:", list(df.columns))

    # Rename to modeling-friendly names
    for src, dst in RENAME_MAP.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")

    # Required cols
    missing = [c for c in ["dt", "atm_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Required column(s) missing: {missing}. "
                         f"Found columns: {list(df.columns)}")

    # Parse dates
    for dcol in ["dt", "reported_dt"]:
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Drop invalid dt/atm_id
    before = len(df)
    df = df.dropna(subset=["dt", "atm_id"]).copy()
    print(f"✅ Dropped {before - len(df)} rows with null dt/atm_id. Remaining: {len(df)}")

    # Normalize dup_flag: drop rows flagged as duplicates (1/true/yes/y)
    if "dup_flag" in df.columns:
        dup_mask = df["dup_flag"].astype(str).str.strip().str.lower().isin(["1", "true", "yes", "y"])
        dropped = int(dup_mask.sum())
        df = df[~dup_mask].copy()
        if dropped:
            print(f"✅ Removed {dropped} duplicate-flagged rows via dup_flag")

    # Coerce numerics and clip to non-negative
    present_num = [c for c in NUMERIC_COLS if c in df.columns]
    for c in present_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        neg = int((df[c] < 0).fillna(False).sum())
        if neg:
            print(f"⚠️  {neg} negative values in {c} -> set to 0")
        df.loc[df[c] < 0, c] = 0.0

    # Quick peek
    print("📏 Rows:", len(df))
    print("🧾 Sample:")
    print(df.head(5).to_string(index=False))

    return df

def aggregate_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    agg = {}
    for c in NUMERIC_COLS:
        if c in df.columns:
            agg[c] = "sum"
    if "region" in df.columns:
        agg["region"] = "first"

    # If no numeric columns exist, just keep first row per (atm_id, dt)
    if not agg:
        agg = "first"

    out = df.groupby(["atm_id", "dt"], as_index=False).agg(agg)
    print(f"🧮 Aggregated duplicates per (atm_id, dt). Rows now: {len(out)}")
    return out

def daily_align_per_atm(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for atm, g in df.groupby("atm_id", sort=False):
        g = g.sort_values("dt")
        if g.empty:
            continue
        idx = pd.date_range(g["dt"].min(), g["dt"].max(), freq="D")
        gg = g.set_index("dt").reindex(idx)
        gg["atm_id"] = atm
        if "region" in gg.columns:
            gg["region"] = gg["region"].ffill().bfill()
        gg.index.name = "dt"
        out.append(gg.reset_index())
    if not out:
        raise ValueError("No data remained after daily alignment.")
    df2 = pd.concat(out, ignore_index=True)
    print(f"📆 Daily aligned per ATM. Rows now: {len(df2)}")
    return df2

def cap_outliers_per_atm(df: pd.DataFrame, q=0.995) -> pd.DataFrame:
    out = df.copy()
    present_num = [c for c in NUMERIC_COLS if c in out.columns]
    for c in present_num:
        caps = out.groupby("atm_id")[c].quantile(q, interpolation="linear").rename(f"{c}_cap")
        out = out.merge(caps, on="atm_id", how="left")
        out[c] = np.where(out[c].notna(), np.minimum(out[c], out[f"{c}_cap"]), out[c])
        out = out.drop(columns=[f"{c}_cap"])
    if present_num:
        print(f"🧯 Outliers capped at {q*100:.1f}th percentile for: {present_num}")
    return out

def add_calendar_features(df: pd.DataFrame, weekend_days) -> pd.DataFrame:
    dts = pd.to_datetime(df["dt"])
    df["dow"] = dts.dt.weekday
    df["is_weekend"] = df["dow"].isin(weekend_days).astype(int)
    df["dom"] = dts.dt.day
    df["month"] = dts.dt.month
    return df

def add_lag_and_moving_averages(df: pd.DataFrame, lags=(1, 7, 14), windows=(7, 14)) -> pd.DataFrame:
    df = df.sort_values(["atm_id", "dt"]).copy()
    present_num = [c for c in NUMERIC_COLS if c in df.columns]
    for t in present_num:
        for L in lags:
            df[f"{t}_lag{L}"] = df.groupby("atm_id")[t].shift(L)
        for W in windows:
            df[f"{t}_ma{W}"] = (
                df.groupby("atm_id")[t]
                .shift(1)  # prevent leakage
                .rolling(W, min_periods=max(3, int(W * 0.6)))
                .mean()
                .reset_index(level=0, drop=True)
            )
    print(f"🧱 Added lags {lags} and MAs {windows} for: {present_num}")
    return df

def build_outputs(input_csv, out_clean, out_features, weekend_arg, fill_feature_nas=True):
    weekend_days = parse_weekend(weekend_arg)

    df = read_and_standardize(input_csv)
    df = aggregate_duplicates(df)
    df = daily_align_per_atm(df)
    df = cap_outliers_per_atm(df, q=0.995)

    # Save clean base
    keep_cols = ["dt", "atm_id"]
    if "region" in df.columns: keep_cols.append("region")
    for c in NUMERIC_COLS:
        if c in df.columns: keep_cols.append(c)
    clean_base = df[keep_cols].copy()
    clean_base.to_csv(out_clean, index=False, encoding="utf-8-sig")
    print(f"💾 Cleaned table -> {out_clean}  (rows={len(clean_base)})")

    # Build features
    feat = add_calendar_features(df.copy(), weekend_days=weekend_days)
    feat = add_lag_and_moving_averages(feat)

    if fill_feature_nas:
        engineered_cols = [c for c in feat.columns if any(s in c for s in ("_lag", "_ma", "dow", "is_weekend", "dom", "month"))]
        feat[engineered_cols] = feat[engineered_cols].fillna(0)

    feat.to_csv(out_features, index=False, encoding="utf-8-sig")
    print(f"💾 Feature matrix -> {out_features}  (rows={len(feat)})")

    # Summary
    try:
        rng = (pd.to_datetime(clean_base["dt"]).min(), pd.to_datetime(clean_base["dt"]).max())
        print(f"🗓️  Date range: {rng[0].date()} .. {rng[1].date()}")
        print(f"🏧 ATMs: {clean_base['atm_id'].nunique()}")
    except Exception:
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=DEFAULT_INPUT, help=f"Path to ATM CSV. Default: {DEFAULT_INPUT}")
    p.add_argument("--out-clean", default=DEFAULT_OUT_CLEAN, help=f"Output CSV for cleaned base. Default: {DEFAULT_OUT_CLEAN}")
    p.add_argument("--out-features", default=DEFAULT_OUT_FEATURES, help=f"Output CSV for features. Default: {DEFAULT_OUT_FEATURES}")
    p.add_argument("--weekend", default=None, help="Comma-separated weekend DOWs (0=Mon..6=Sun). Default: '4,5'")
    p.add_argument("--no-impute", action="store_true", help="Do NOT impute NaNs in engineered features.")
    args = p.parse_args()

    try:
        build_outputs(
            input_csv=args.input,
            out_clean=args.out_clean,
            out_features=args.out_features,
            weekend_arg=args.weekend,
            fill_feature_nas=not args.no_impute,
        )
    except Exception as ex:
        eprint("\n❌ ERROR:", ex)
        eprint("👉 Tips:")
        eprint("  • Check the file path: --input should point to your CSV.")
        eprint("  • Ensure columns include at least: dt, atm_id (case-insensitive).")
        eprint("  • If your headers differ, adjust RENAME_MAP or your CSV columns.")
        eprint("  • If you passed --weekend, use digits 0..6 like --weekend \"4,5\".")
        sys.exit(1)

if __name__ == "__main__":
    main()
