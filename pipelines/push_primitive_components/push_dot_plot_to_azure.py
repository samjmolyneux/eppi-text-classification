import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

create_dot_plot = "./../primitive_components/create_dot_plot"

create_dot_plot_component = command(
    name="shap_dot_plot_for_classifier_workbench",
    display_name="SHAP dot plot for classifier workbench",
    description=("Create a SHAP dot plot for classifier workbench"),
    inputs={
        "shap_values": Input(type="uri_folder"),
        "X": Input(type="uri_folder"),
        "feature_names": Input(type="uri_folder"),
        "num_display": Input(type="uri_folder"),
        "log_scale": Input(type="uri_folder"),
        "plot_zero": Input(type="uri_folder"),
    },
    outputs={
        "dot_plot": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=create_dot_plot,
    command="""python create_dot_plot.py \
            --shap_values ${{inputs.shap_values}} \
            --X ${{inputs.X}} \
            --feature_names ${{inputs.feature_names}} \
            --num_display ${{inputs.num_display}} \
            --log_scale ${{inputs.log_scale}} \
            --plot_zero ${{inputs.plot_zero}} \
            --dot_plot ${{outputs.dot_plot}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
create_dot_plot_component = ml_client.create_or_update(
    create_dot_plot_component.component
)

# Create (register) the component in your workspace
print(
    f"Component {create_dot_plot_component.name} with Version {create_dot_plot_component.version} is registered"
)
