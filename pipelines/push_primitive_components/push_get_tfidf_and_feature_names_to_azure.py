import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)
get_tfidf_and_feature_names = "./../primitive_components/get_tfidf_and_feature_names"


get_tfidf_and_feature_names_component = command(
    name="get_tfidf_and_feature_names",
    display_name="Get tfidf and feature names classifier workbench",
    description="Get tfidf and feature names for eppi classifier workbench",
    inputs={
        "data": Input(type="uri_file"),
        "title_header": Input(type="uri_file"),
        "abstract_header": Input(type="uri_file"),
    },
    outputs={
        "feature_names": Output(type="uri_folder", mode="rw_mount"),
        "tfidf_scores": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=get_tfidf_and_feature_names,
    command="""python get_tfidf_and_feature_names.py \
            --data ${{inputs.data}} \
            --title_header ${{inputs.title_header}} \
            --abstract_header ${{inputs.abstract_header}} \
            --tfidf_scores ${{outputs.tfidf_scores}} \
            --feature_names ${{outputs.feature_names}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
get_tfidf_and_feature_names_component = ml_client.create_or_update(
    get_tfidf_and_feature_names_component.component
)

# Create (register) the component in your workspace
print(
    f"Component {get_tfidf_and_feature_names_component.name} with Version {get_tfidf_and_feature_names_component.version} is registered"
)
