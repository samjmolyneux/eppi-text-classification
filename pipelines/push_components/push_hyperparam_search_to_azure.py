import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

hyperparameter_search = "./components/hyperparameter_search"

hyperparameter_search_component = command(
    name="hyperparameter_search_for_classifier_workbench",
    display_name="Hyperparameter search for eppi classifier workbench",
    description=(
        "Uses parallel optuna to search for best hyperparameters for a given "
        "model, storing the history on a sqlite database"
    ),
    inputs={
        "labels": Input(type="uri_folder"),
        "tfidf_scores": Input(type="uri_folder"),
        "search_parameters": Input(type="uri_file"),
    },
    outputs={
        "best_params": Output(type="uri_folder", mode="rw_mount"),
        "search_db": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=hyperparameter_search,
    command="""python optuna_search.py \
            --labels ${{inputs.labels}} \
            --tfidf_scores ${{inputs.tfidf_scores}} \
            --search_parameters ${{inputs.search_parameters}} \
            --best_params ${{outputs.best_params}} \
            --search_db ${{outputs.search_db}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
hyperparameter_search_component = ml_client.create_or_update(
    hyperparameter_search_component.component
)

# Create (register) the component in your workspace
print(
    f"Component {hyperparameter_search_component.name} with Version {hyperparameter_search_component.version} is registered"
)
