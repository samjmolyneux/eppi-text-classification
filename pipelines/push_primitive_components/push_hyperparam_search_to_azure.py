import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_temp_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_temp_env(ml_client)

hyperparameter_search = "./../primitive_components/hyperparameter_search"

hyperparameter_search_component = command(
    name="hyperparameter_search_for_classifier_workbench_single_instance",
    display_name="Hyperparameter search for eppi classifier workbench",
    description=(
        "Uses parallel optuna to search for best hyperparameters for a given "
        "model, storing the history on a sqlite database"
    ),
    inputs={
        "labels": Input(
            type="uri_folder",
        ),
        "tfidf_scores": Input(
            type="uri_folder",
        ),
        "model_name": Input(
            type="string",
        ),
        "n_folds": Input(
            type="integer",
            default=3,
            min=2,
        ),
        "num_cv_repeats": Input(
            type="integer",
            default=1,
            min=1,
        ),
        "timeout": Input(
            type="integer",
            default=None,
        ),
        "use_early_terminator": Input(
            type="boolean",
            default=False,
        ),
        "use_worse_than_first_two_pruner": Input(
            type="boolean",
            default=False,
        ),
        "max_n_search_iterations": Input(
            type="integer",
            optional=True,
        ),
        "max_stagnation_iterations": Input(
            type="integer",
            default=None,
            optional=True,
        ),
        "wilcoxon_trial_pruner_threshold": Input(
            type="number",
            default=None,
            optional=True,
        ),
        "resume_search_db": Input(
            type="uri_file",
            optional=True,
        ),
        "user_selected_hyperparameter_ranges": Input(
            type="string",
            default=None,
            optional=True,
        ),
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
            --n_folds ${{inputs.n_folds}} \
            --num_cv_repeats ${{inputs.num_cv_repeats}} \
            --timeout ${{inputs.timeout}} \
            --use_early_terminator ${{inputs.use_early_terminator}} \
            --use_worse_than_first_two_pruner ${{inputs.use_worse_than_first_two_pruner}} \
            $[[--max_n_search_iterations ${{inputs.max_n_search_iterations}}]] \
            $[[--max_stagnation_iterations ${{inputs.max_stagnation_iterations}}]] \
            $[[--wilcoxon_trial_pruner_threshold ${{inputs.wilcoxon_trial_pruner_threshold}}]] \
            $[[--resume_search_db ${{inputs.resume_search_db}}]] \
            $[[--user_selected_hyperparameter_ranges ${{inputs.user_selected_hyperparameter_ranges}}]] \
            --best_params ${{outputs.best_params}} \
            --search_db ${{outputs.search_db}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
    # compute="sams-f16-v2"
)

# Now we register the component to the workspace
hyperparameter_search_component = ml_client.create_or_update(
    hyperparameter_search_component.component, version="prim_1.18"
)

# Create (register) the component in your workspace
print(
    f"Component {hyperparameter_search_component .name} with Version {hyperparameter_search_component .version} is registered"
)
