import boto3
import sagemaker
from sagemaker.sklearn import SKLearn
import time

aws_region = "us-east-1"   
s3_bucket = "mlonawsarun0710"  
script_file_name = "modeltrainonaws.py"
script_s3_path = f"project1/scripts/{script_file_name}" 
training_input_path = f"s3://{s3_bucket}/project1/data/train_data/train_iris.csv"  # S3 path to training data
output_path = f"s3://{s3_bucket}/project1/output/"  # S3 path for output artifacts

role = "arn:aws:iam::211125398648:role/MLOps" 

s3 = boto3.client('s3')
sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))

def upload_script_to_s3(script_file):
    print(f"Uploading {script_file} to s3://{s3_bucket}/{script_s3_path}")
    s3.upload_file(script_file, s3_bucket, script_s3_path)
    print("Script uploaded successfully.")

def run_sagemaker_processing_job():
    # Define the ScriptProcessor with Python runtime
    sk_estimator = SKLearn(
    entry_point="modeltrainonaws.py",  # Training script filename
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    framework_version="0.23-1",
    py_version="py3",
    output_path=output_path,
    )
    
    sk_estimator.fit({"train": training_input_path})

if __name__ == "__main__":
    # Ensure script is uploaded, then run the processing job
    upload_script_to_s3(script_file_name)
    run_sagemaker_processing_job()
