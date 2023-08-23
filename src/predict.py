import sys
import pandas as pd
import pickle

# Taking two arguments model and file with data for predict

if len(sys.argv) != 3:
    print("USAGE: python predict.py <model> <file>")

else:
    model_path = sys.argv[1]
    file_path = sys.argv[2]

    # reading a model
    print("Loading model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded!")

    # Predictions for each row of argument file
    print("Making Predictions...")
    df = pd.read_parquet(file_path)
    df["total_sales"] = model.predict(df)
    print(df.head())

    # Save predictions to a new file
    print("Saving...")
    df.to_parquet("../data/predict-done-2023-08-03.parquet", index=False)
    print("Saved!")


