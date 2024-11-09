import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import time

aws_region = "us-east-1"   
s3_bucket = "mlonawsarun0710"  
script_file_name = "datapreponaws.py"
script_s3_path = f"project1/scripts/{script_file_name}"  # Ensure this is the correct relative path in S3

role = "arn:aws:iam::211125398648:role/MLOps" 
input_bucket = "mlonawsarun0710"
input_path = "data/originaldata.csv"
output_bucket = "mlonawsarun0710"
output_path = "data/input/"

s3 = boto3.client('s3')
sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))

def upload_script_to_s3(script_file):
    print(f"Uploading {script_file} to s3://{s3_bucket}/{script_s3_path}")
    s3.upload_file(script_file, s3_bucket, script_s3_path)
    print("Script uploaded successfully.")

def run_sagemaker_processing_job():
    # Define the ScriptProcessor with Python runtime
    sklearn_processor = SKLearnProcessor(
    framework_version="0.20.0",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    )

    # Run the processing job
    print("Starting SageMaker processing job...")
    sklearn_processor.run(
        code="s3://mlonawsarun0710/project1/scripts/datapreponaws.py",  # Correct: Provide S3 path here
        inputs=[
        ProcessingInput(source="s3://mlonawsarun0710/project1/data/originaldata.csv", destination="/opt/ml/processing/input"),
    ],
        outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train", destination="s3://mlonawsarun0710/project1/data/train_data"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",destination="s3://mlonawsarun0710/project1/data/test_data"),
    ]
    )
    

if __name__ == "__main__":
    # Ensure script is uploaded, then run the processing job
    upload_script_to_s3(script_file_name)
    run_sagemaker_processing_job()
