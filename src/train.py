import pandas as pd
import pickle
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Construct a python script src/train.py that receives as argument 
# the path of preprocessed training file:

def train(pre_train_file_path):
    df = pd.read_parquet(pre_train_file_path)

    X = df.drop(['total_sales'], axis=1)
    y = df['total_sales']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, random_state=195)  
    model.fit(X_train, y_train)

    return model

if __name__ == "__init__":
    if len(sys.argv) != 2:
        print("USAGE: python train.py <pre_train_file_path>")
    else:
        pre_train_file_path = sys.argv[1]
        print("Training model...")
        model = train(pre_train_file_path)
        model_path = "../models/model-2023-08-01.pickle"
        print("Model trained!")
        print(f"Saving to {model_path} file...")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        print("Model saved!")
