import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

calculate_shap_values = "./../primitive_components/calculate_shap_values"

calculate_shap_values_component = command(
    name="calculate_shap_values_for_classifier_workbench",
    display_name="Calculate SHAP values for classifier workbench",
    description=("Creates a shap plotter object and calculates the shap values"),
    inputs={
        "model": Input(type="uri_folder"),
        "X": Input(type="uri_folder"),
        "feature_names": Input(type="uri_folder"),
    },
    outputs={
        "shap_values": Output(type="uri_folder", mode="rw_mount"),
        "shap_expected_value": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=calculate_shap_values,
    command="""python calculate_shap_values.py \
            --model ${{inputs.model}} \
            --X ${{inputs.X}} \
            --feature_names ${{inputs.feature_names}} \
            --shap_values ${{outputs.shap_values}} \
            --shap_expected_value ${{outputs.shap_expected_value}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

calculate_shap_values_component = ml_client.create_or_update(
    calculate_shap_values_component.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {calculate_shap_values_component.name} with Version {calculate_shap_values_component.version} is registered"
)
