import json
import os
import tarfile
import pandas as pd
#import joblib
from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

if __name__ == "__main__":
    model_path = os.path.join("/opt/ml/processing/model", "model.tar.gz")
    print(f"Extracting model from path: {model_path}")
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")
    print("Loading model")
    model = joblib.load("lrmodeliris.joblib")

    print("Loading test input data")
    test_data = os.path.join("/opt/ml/processing/test", "test_iris.csv")
    df = pd.read_csv(test_data, header=None)
    
    X_test = df.iloc[:, 1:]
    y_test = df.iloc[:, 0]
    
    predictions = model.predict(X_test)

    print("Creating classification evaluation report")
    report_dict = classification_report(y_test, predictions, output_dict=True)
    report_dict["accuracy"] = accuracy_score(y_test, predictions)

    print(f"Classification report:\n{format(report_dict)}")

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print(f"Saving classification report to {format(evaluation_output_path)}")

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))