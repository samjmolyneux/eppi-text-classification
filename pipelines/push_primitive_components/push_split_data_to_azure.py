import os
from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

split_data = "./../primitive_components/split_data"

split_data_component = command(
    name="split_data_for_classifier_workbench",
    display_name="Split data into two sets",
    description=(
        "Uses train_test_split to split the data into two sets, storing the split data"
    ),
    inputs={
        "labels": Input(type="uri_folder"),
        "tfidf_scores": Input(type="uri_folder"),
        "test_size": Input(type="number"),
    },
    outputs={
        "X_train": Output(type="uri_folder", mode="rw_mount"),
        "X_test": Output(type="uri_folder", mode="rw_mount"),
        "y_train": Output(type="uri_folder", mode="rw_mount"),
        "y_test": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=split_data,
    command="""python split_data.py \
            --labels ${{inputs.labels}} \
            --tfidf_scores ${{inputs.tfidf_scores}} \
            --test_size ${{inputs.test_size}} \
            --X_train ${{outputs.X_train}} \
            --X_test ${{outputs.X_test}} \
            --y_train ${{outputs.y_train}} \
            --y_test ${{outputs.y_test}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
split_data_component = ml_client.create_or_update(
    split_data_component.component, version="prim_1.1"
)

# Create (register) the component in your workspace
print(
    f"Component {split_data_component.name} with Version {split_data_component.version} is registered"
)
