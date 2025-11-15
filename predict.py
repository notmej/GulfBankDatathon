# predict.py
import predictNaive
import predictMovingAvrg
import predictExpSmooth

if __name__ == "__main__":
    print("=== PREDICTING: Naive Model ===")
    predictNaive.main()            # writes predictions_naive.csv

    print("=== PREDICTING: Moving Average Model ===")
    predictMovingAvrg.main()       # writes predictionsMovingAvrg.csv

    print("=== PREDICTING: Exponential Smoothing Model ===")
    predictExpSmooth.main()        # writes predictionsExpSmooth.csv

    print("✅ All model predictions generated successfully.")
