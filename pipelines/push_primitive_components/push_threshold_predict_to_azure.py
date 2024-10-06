import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

threshold_predict = "./../primitive_components/threshold_predict"

threshold_predict = command(
    name="predict_given_threshold_for_classifier_workbench",
    display_name="Predict given a threshold for classifier workbench model",
    description=("Predict given a threshold for classifier workbench model"),
    inputs={
        "X": Input(type="uri_folder"),
        "model": Input(type="uri_folder"),
        "threshold": Input(type="uri_folder"),
    },
    outputs={
        "y_pred": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=threshold_predict,
    command="""python threshold_predict.py \
            --X ${{inputs.X}} \
            --model ${{inputs.model}} \
            --threshold ${{inputs.threshold}} \
            --y_pred ${{outputs.y_pred}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
threshold_predict = ml_client.create_or_update(
    threshold_predict.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {threshold_predict.name} with Version {threshold_predict.version} is registered"
)
