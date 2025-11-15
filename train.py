# train.py
import dataCleaning  # <-- add this line
import trainNaive
import trainMovingAvrg
import trainExpSmooth

if __name__ == "__main__":
    print("=== CLEANING DATA ===")
    # run the cleaner from dataCleaning.py
    # adjust the arguments if your cleaner expects different ones
    dataCleaning.build_outputs(
        input_csv="atm_transactions_train.csv",   # input raw training data
        out_clean="atm_transactions_train_clean.csv",  # cleaned dataset
        out_features="features.csv",              # optional features file
        weekend_arg="4,5",                        # Friday/Saturday weekend
        fill_feature_nas=True
    )
    print("✅ Data cleaned successfully. Output: atm_transactions_train_clean.csv")

    # Now run your model trainings using the cleaned file
    print("\n=== TRAINING: Naive Model ===")
    trainNaive.main()           # produces model_naive_params.csv

    print("\n=== TRAINING: Moving Average Model ===")
    trainMovingAvrg.main()      # produces modelMovingAvrg_params.csv

    print("\n=== TRAINING: Exponential Smoothing Model ===")
    trainExpSmooth.main()       # produces modelExpSmooth_params.csv

    print("\n✅ All models trained successfully.")
