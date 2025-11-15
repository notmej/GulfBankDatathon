import pandas as pd

# Load your best model's predictions (here, Exponential Smoothing)
df = pd.read_csv("predictionsExpSmooth.csv")

# Keep only the required columns and order
df = df[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]]

# Save as final submission file
df.to_csv("predictions.csv", index=False)

print("✅ Created predictions.csv successfully:", df.shape)
