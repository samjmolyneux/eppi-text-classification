import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

get_labels_file = "./../primitive_components/get_labels"


get_labels_component = command(
    name="get_labels_for_classifier_workbench",
    display_name="Get labels for classifier workbench",
    description=(
        "Given a column name and positive class, create a binary np array of labels"
    ),
    inputs={
        "data": Input(type="uri_file"),
        "label_column_name": Input(type="string", default="included"),
        "positive_class_value": Input(type="string", default="1"),
    },
    outputs={
        "labels": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=get_labels_file,
    command="""python get_labels.py \
            --data ${{inputs.data}} \
            --label_column_name ${{inputs.label_column_name}} \
            --positive_class_value ${{inputs.positive_class_value}} \
            --labels ${{outputs.labels}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
get_labels_component = ml_client.create_or_update(
    get_labels_component.component, version="prim_1.2"
)

# Create (register) the component in your workspace
print(
    f"Component {get_labels_component.name} with Version {get_labels_component.version} is registered"
)
