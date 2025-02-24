import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

predict_scores = "./../primitive_components/predict_scores"

predict_probabilities_component = command(
    name="predict_probabilities_for_eppi_classifier_workbench",
    display_name="predict_probabilities_for_eppi_classifier_workbench",
    description=(
        "Takes a model from the eppi classifier workbench and uses it to predict "
        "probabilities"
    ),
    inputs={
        "X": Input(type="uri_folder"),
        "model": Input(type="uri_folder"),
    },
    outputs={
        "y_pred_probs": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=predict_scores,
    command="""python predict_scores.py \
            --X ${{inputs.X}} \
            --model ${{inputs.model}} \
            --y_pred_probs ${{outputs.y_pred_probs}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
predict_probabilities_component = ml_client.create_or_update(
    predict_probabilities_component.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {predict_probabilities_component.name} with Version {predict_probabilities_component.version} is registered"
)
