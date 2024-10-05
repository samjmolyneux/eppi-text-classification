import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

plotly_confusion = "./components/plotly_confusion"

plotly_confusion_component = command(
    name="confusion_plot_for_classifier_workbench",
    display_name="Confusion Matrix Plot",
    description=(
        "Confusion matrix that plots three or two confusion plots based on whether"
        "test data is provided"
    ),
    inputs={
        "y_train": Input(type="uri_folder"),
        "y_train_pred": Input(type="uri_folder"),
        "y_val": Input(type="uri_folder"),
        "y_val_pred": Input(type="uri_folder"),
        "y_test": Input(type="uri_folder", optional=True),
        "y_test_pred": Input(type="uri_folder", optional=True),
    },
    outputs={
        "confusion_plot": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=plotly_confusion,
    command="""python plotly_confusion.py \
            --y_train ${{inputs.y_train}} \
            --y_train_pred ${{inputs.y_train_pred}} \
            --y_val ${{inputs.y_val}} \
            --y_val_pred ${{inputs.y_val_pred}} \
            $[[--y_test ${{inputs.y_test}}]] \
            $[[--y_test_pred ${{inputs.y_test_pred}}]] \
            --confusion_plot ${{outputs.confusion_plot}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
plotly_confusion_component = ml_client.create_or_update(
    plotly_confusion_component.component
)

# Create (register) the component in your workspace
print(
    f"Component {plotly_confusion_component.name} with Version {plotly_confusion_component.version} is registered"
)
