import os

from azure.ai.ml.entities import Environment
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

dependencies_dir = "./../dependencies"

custom_env_name = "eppi-classifier-workbench-env"

pipeline_job_env = Environment(
    name=custom_env_name,
    description="Python environment for eppi classifier workbench.",
    conda_file=os.path.join(dependencies_dir, "eppi-classifier-workbench-env.yaml"),
    image="mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest",
    version="1.0.0",
)
pipeline_job_env = ml_client.environments.create_or_update(pipeline_job_env)

print(
    f"Environment with name {pipeline_job_env.name} is registered to workspace, the environment version is {pipeline_job_env.version}"
)
