"""Loading data from S3 and saving back the model artifacts """

import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model():
    input_data_path = os.path.join("/opt/ml/input/data/train", "train_iris.csv")
    #input_data_path = "data/iris_train.csv"
    df = pd.read_csv(input_data_path, header=None)
    X_train = df.iloc[:, 1:]
    y_train = df.iloc[:, 0]
    model = LogisticRegression(solver='liblinear')
    print("Training LR model")
    model.fit(X_train, y_train)
    model_output_directory = os.path.join("/opt/ml/model", "lrmodeliris.joblib")
    #model_output_directory = os.path.join("data", "lrmodeliris.joblib")
    print(f"Saving model to {model_output_directory}")
    joblib.dump(model, model_output_directory)
    
if __name__ == "__main__":
    train_model()