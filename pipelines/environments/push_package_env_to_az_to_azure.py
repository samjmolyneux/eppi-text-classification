import os

from azure.ai.ml.entities import Environment
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

dependencies_dir = "./dependencies"

custom_env_name = "aml-eppi-text-classification"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Custom environment for eppi classifier workbench pipeline",
    conda_file=os.path.join(dependencies_dir, "conda.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1.3",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)
