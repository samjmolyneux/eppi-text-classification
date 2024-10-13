import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

train_model = "./../primitive_components/train_model"

train_model_component = command(
    name="fit_model_for_classifier_workbench",
    display_name="fit_model_for_classifier_workbench",
    description=(
        "Trains a model for classifier workbench using given data and model parameters"
    ),
    inputs={
        "X_train": Input(type="uri_folder"),
        "y_train": Input(type="uri_folder"),
        "model_parameters": Input(type="uri_file"),
    },
    outputs={
        "model": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=train_model,
    command="""python train_model.py \
            --X_train ${{inputs.X_train}} \
            --y_train ${{inputs.y_train}} \
            --model_parameters ${{inputs.model_parameters}} \
            --model ${{outputs.model}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
train_model_component = ml_client.create_or_update(
    train_model_component.component, version="prim_1.1"
)

# Create (register) the component in your workspace
print(
    f"Component {train_model_component.name} with Version {train_model_component.version} is registered"
)
