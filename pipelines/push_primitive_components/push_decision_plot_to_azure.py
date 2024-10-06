import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

create_decision_plot = "./../primitive_components/create_decision_plot"

create_decision_plot_component = command(
    name="create_shap_decision_plot_for_classifier_workbench",
    display_name="SHAP decision plot for classifier workbench",
    description=("Create a SHAP decision plot for classifier workbench"),
    inputs={
        "expected_shap_value": Input(type="uri_folder"),
        "threshold": Input(type="uri_folder"),
        "shap_values": Input(type="uri_folder"),
        "X": Input(type="uri_folder"),
        "feature_names": Input(type="uri_folder"),
        "num_display": Input(type="integer", default=10),
        "log_scale": Input(type="boolean", default=False),
    },
    outputs={
        "decision_plot": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=create_decision_plot,
    command="""python create_decision_plot.py \
            --expected_shap_value ${{inputs.expected_shap_value}} \
            --threshold ${{inputs.threshold}} \
            --shap_values ${{inputs.shap_values}} \
            --X ${{inputs.X}} \
            --feature_names ${{inputs.feature_names}} \
            --num_display ${{inputs.num_display}} \
            --log_scale ${{inputs.log_scale}} \
            --decision_plot ${{outputs.decision_plot}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

create_decision_plot_component = ml_client.create_or_update(
    create_decision_plot_component.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {create_decision_plot_component .name} with Version {create_decision_plot_component .version} is registered"
)
