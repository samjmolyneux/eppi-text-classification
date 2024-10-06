import os

from azure.ai.ml.entities import Environment
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

dependencies_dir = "./dependencies"

custom_env_name = "display-image-env"

display_image_env = Environment(
    name=custom_env_name,
    description="Environment for displaying images in azure ml",
    conda_file=os.path.join(dependencies_dir, "display_image_env.yaml"),
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    version="0.1.0",
)
display_image_env = ml_client.environments.create_or_update(display_image_env)

print(
    f"Environment with name {display_image_env.name} is registered to workspace, the environment version is {display_image_env.version}"
)
