# predict.py
import predictNaive
import predictMovingAvrg
import predictExpSmooth
import pandas as pd


if __name__ == "__main__":
    print("=== PREDICTING: Naive Model ===")
    predictNaive.main()            # writes predictions_naive.csv

    print("=== PREDICTING: Moving Average Model ===")
    predictMovingAvrg.main()       # writes predictionsMovingAvrg.csv

    print("=== PREDICTING: Exponential Smoothing Model ===")
    predictExpSmooth.main()        # writes predictionsExpSmooth.csv

    print("✅ All model predictions generated successfully.")

    # Use your best model’s predictions (e.g. Exponential Smoothing)
    df = pd.read_csv("predictionsExpSmooth.csv")[["dt", "atm_id", "predicted_withdrawn_kwd", "predicted_withdraw_count"]]
    df.to_csv("predictions.csv", index=False)
    print("✅ Final submission file created: predictions.csv")