import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

hyperparameter_search = "./../primitive_components/hyperparameter_search"

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
        "model_name": Input(type="string"),
        "num_trials_per_job": Input(type="integer"),
        "n_folds": Input(type="integer", default=3),
        "num_cv_repeats": Input(type="integer", default=1),
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
            --model_name ${{inputs.model_name}} \
            --num_trials_per_job ${{inputs.num_trials_per_job}} \
            --n_folds ${{inputs.n_folds}} \
            --num_cv_repeats ${{inputs.num_cv_repeats}} \
            --best_params ${{outputs.best_params}} \
            --search_db ${{outputs.search_db}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
hyperparameter_search_component = ml_client.create_or_update(
    hyperparameter_search_component.component, version="prim_1.2"
)

# Create (register) the component in your workspace
print(
    f"Component {hyperparameter_search_component .name} with Version {hyperparameter_search_component .version} is registered"
)
