import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

get_threshold = "./../primitive_components/get_threshold"

get_threshold_component = command(
    name="get_classification_threshold_for_classifier_workbench",
    display_name="Get the classification threshold for a given TPR",
    description=(
        "For a given desired true positive rate, get the classification threshold"
    ),
    inputs={
        "y": Input(type="uri_folder"),
        "X": Input(type="uri_folder"),
        "model": Input(type="uri_folder"),
        "target_tpr": Input(type="string"),
    },
    outputs={
        "threshold": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=get_threshold,
    command="""python get_threshold.py \
            --y ${{inputs.y}} \
            --X ${{inputs.X}} \
            --model ${{inputs.model}} \
            --target_tpr ${{inputs.target_tpr}} \
            --threshold ${{outputs.threshold}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
get_threshold_component = ml_client.create_or_update(
    get_threshold_component.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {get_threshold_component.name} with Version {get_threshold_component.version} is registered"
)
