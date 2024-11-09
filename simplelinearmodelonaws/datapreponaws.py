# Here we assume that iris data is already there in s3 bucket. So, we will first write a python code which will read data from s3 and do 
# some basic data pre processing and store the refined output in s3 directly.

"""
#First let us load the sample data from local machine and do data prep. This part of the code will get commented.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data/originaldata.csv")

df = df.dropna()
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:, 1:])
df_scaled = pd.DataFrame(scaled_features, columns=df.columns[1:])
df_scaled.insert(0, df.columns[0], df.iloc[:, 0].values)


train_df, test_df = train_test_split(
    df_scaled, test_size=0.33, random_state=42, stratify=df["class"]
)

os.makedirs("data/input", exist_ok=True)

# Save the training DataFrame to a CSV file without index or header
train_df.to_csv("data/input/iris_train.csv", index=False, header=None)

# Save the testing DataFrame to a CSV file without index or header
test_df.to_csv("data/input/iris_test.csv", index=False, header=None)
"""


### Data prepcode to run on aws sagemaker

"""Load data from S3, clean, transform, and save it back to S3."""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import argparse

def data_preparation():
    input_data_path = os.path.join("/opt/ml/processing/input", "originaldata.csv")
    df = pd.read_csv(input_data_path)
    
    df = df.dropna()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.iloc[:, 1:])
    df_scaled = pd.DataFrame(scaled_features, columns=df.columns[1:])
    df_scaled.insert(0, df.columns[0], df.iloc[:, 0].values)
    
    train_df, test_df = train_test_split( df_scaled, test_size=0.33, random_state=42, stratify=df["class"] )
    
    train_path = os.path.join("/opt/ml/processing/train", "train_iris.csv")
    test_path = os.path.join("/opt/ml/processing/test", "test_iris.csv")
    
    train_df.to_csv(train_path, header=False, index=False)
    test_df.to_csv(test_path, header=False, index=False)
    
if __name__ == "__main__":
    data_preparation()