import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

plotly_roc = "./components/plotly_roc"

plotly_roc_component = command(
    name="roc_plot_for_eppi_classifier_workbench",
    display_name="ROC plot for eppi classifier workbench",
    description=("Plots ROC curve for given labels and predicted probabilities"),
    inputs={
        "y": Input(type="uri_folder"),
        "y_pred_probs": Input(type="uri_folder"),
    },
    outputs={
        "roc_plot": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=plotly_roc,
    command="""python plotly_roc.py \
            --y ${{inputs.y}} \
            --y_pred_probs ${{inputs.y_pred_probs}} \
            --roc_plot ${{outputs.roc_plot}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
plotly_roc_component = ml_client.create_or_update(plotly_roc_component.component)

# Create (register) the component in your workspace
print(
    f"Component {plotly_roc_component.name} with Version {plotly_roc_component.version} is registered"
)
