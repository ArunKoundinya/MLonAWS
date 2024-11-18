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