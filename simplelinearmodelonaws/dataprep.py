# Import necessary libraries
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
import os
import boto3

# Load the Iris dataset from sklearn
iris = datasets.load_iris()

# Create a DataFrame using the iris data and feature names
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add a column for the target classes
df["class"] = pd.Series(iris.target)

# Reorder the columns to place the 'class' column first
cols = list(df.columns)
cols = [cols[-1]] + cols[:-1]  # Move 'class' to the front
df = df[cols]

# Split the DataFrame into training and testing sets
train_df, test_df = train_test_split(
    df, test_size=0.33, random_state=42, stratify=df["class"]
)

# Prepare an inference DataFrame by dropping the 'class' column from the test set
infer_df = test_df.drop(columns=["class"])

# Create a directory named 'data' if it doesn't already exist
os.makedirs("basicxgboostmodel/data", exist_ok=True)

# Save the training DataFrame to a CSV file without index or header
train_df.to_csv("basicxgboostmodel/data/iris_train.csv", index=False, header=None)

# Save the testing DataFrame to a CSV file without index or header
test_df.to_csv("basicxgboostmodel/data/iris_test.csv", index=False, header=None)

# Save the inference DataFrame to a CSV file without index or header
infer_df.to_csv("basicxgboostmodel/data/iris_infer.csv", index=False, header=None)

# Save the original DataFrame (with the class column) to a CSV file
df.to_csv("basicxgboostmodel/data/originaldata.csv", index=False)

# Initialize a boto3 client for S3
s3 = boto3.client('s3')

# Specify the S3 bucket name and path where the files will be uploaded
s3_bucket_name = "mlonawsarun0710"
s3_bucket_path = "project1/data/"

# List of files to upload to S3
files_to_upload = [
    "basicxgboostmodel/data/iris_train.csv",
    "basicxgboostmodel/data/iris_test.csv",
    "basicxgboostmodel/data/iris_infer.csv",
    "basicxgboostmodel/data/originaldata.csv"
]

# Loop through the list of files and upload each one to S3
for file in files_to_upload:
    # Upload the file to the specified S3 bucket and path
    s3.upload_file(file, s3_bucket_name, s3_bucket_path + os.path.basename(file))
