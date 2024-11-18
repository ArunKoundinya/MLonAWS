import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn import SKLearn
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput


aws_region = "us-east-1"   
role = "arn:aws:iam::211125398648:role/MLOps" 

sagemaker_client = boto3.client("sagemaker", region_name=aws_region)
session = sagemaker.Session(boto_session=boto3.Session(region_name=aws_region))

def run_sagemaker_pipeline():
    """ Pre Processing """
    sklearn_processor = SKLearnProcessor(
        framework_version="0.20.0",
        role=role,
        instance_type="ml.t3.medium",
        instance_count=1,
    )
    
    processing_step = ProcessingStep(
        name="DataPreprocessingStep",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source="s3://mlonawsarun0710/project1/data/originaldata.csv",
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train", destination="s3://mlonawsarun0710/project1/data/train_data"),
            ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test",destination="s3://mlonawsarun0710/project1/data/test_data"),        
        ],
        code="data_processing_script.py" 
    )
    
    """ Model Training """
    sk_estimator = SKLearn(
        entry_point="training_model_script.py",
        role=role,
        instance_count=1,
        instance_type="ml.m5.large",
        framework_version="0.20.0",
        py_version="py3",
        output_path="s3://mlonawsarun0710/project1/model-artifacts/",
    )
    
    training_step = TrainingStep(
        name="ModelTrainingStep",
        estimator=sk_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            )
        },
    )

    """ Evaluation """
    evaluation_step = ProcessingStep(
        name="EvaluationStep",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs[
                    "test_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination="s3://mlonawsarun0710/project1/evaluation"),
        ],
        code="evaluate_model_script.py" 
    )

    """ Pipeline Definition """
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[processing_step, training_step, evaluation_step]
    )
    
    pipeline.upsert(role_arn=role)
    print("Pipeline created or updated successfully.")
    
    execution = pipeline.start()
    print(f"Pipeline execution started. Execution ARN: {execution.arn}")

if __name__ == "__main__":
    run_sagemaker_pipeline()
