import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

code_path = "../primitive_components/find_single_model"

component_command = command(
    name="find_single_model_for_classifier_workbench",
    display_name="Find single model for eppi classifier workbench",
    description=(
        "Takes tsv, performs tokenisation, tfidf, and "
        "searches for the best hyperparameters for a given model."
    ),
    inputs={
        "data": Input(type="uri_folder"),
        "title_header": Input(type="string", default="PaperTitle", optional=True),
        "abstract_header": Input(type="string", default="Abstract", optional=True),
        "label_header": Input(type="string", default="Label", optional=True),
        "positive_class_value": Input(type="string", default="1", optional=True),
        "model_name": Input(type="string"),
        "hyperparameter_search_ranges": Input(type="uri_folder"),
        "max_n_search_iterations": Input(type="integer", optional=True),
        "nfolds": Input(type="integer", default=3, optional=True),
        "num_cv_repeats": Input(type="integer", default=1, optional=True),
        "timeout": Input(type="integer", optional=True),
        "use_early_terminator": Input(type="boolean", default=False, optional=True),
        "max_stagnation_iterations": Input(type="integer", optional=True),
        "wilcoxon_trial_pruner_threshold": Input(type="number", optional=True),
        "use_worse_than_first_two_pruner": Input(
            type="boolean", default=False, optional=True
        ),
        "study_name": Input(type="string", default="hyperparam_search", optional=True),
    },
    outputs={
        "plots": Output(type="uri_folder", mode="rw_mount"),
        "search_db": Output(type="uri_folder", mode="rw_mount"),
        "feature_names": Output(type="uri_folder", mode="rw_mount"),
        "tfidf_scores": Output(type="uri_folder", mode="rw_mount"),
        "labels": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=code_path,
    command=(
        "python find_single_model.py "
        "--data ${{inputs.data}} "
        "--model_name ${{inputs.model_name}} "
        "--hyperparameter_search_ranges ${{inputs.hyperparameter_search_ranges}} "
        "$[[--title_header ${{inputs.title_header}}]] "
        "$[[--abstract_header ${{inputs.abstract_header}}]] "
        "$[[--label_header ${{inputs.label_header}}]] "
        "$[[--positive_class_value ${{inputs.positive_class_value}}]] "
        "$[[--max_n_search_iterations ${{inputs.max_n_search_iterations}}]] "
        "$[[--nfolds ${{inputs.nfolds}}]] "
        "$[[--num_cv_repeats ${{inputs.num_cv_repeats}}]] "
        "$[[--timeout ${{inputs.timeout}}]] "
        "$[[--use_early_terminator ${{inputs.use_early_terminator}}]] "
        "$[[--max_stagnation_iterations ${{inputs.max_stagnation_iterations}}]] "
        "$[[--wilcoxon_trial_pruner_threshold ${{inputs.wilcoxon_trial_pruner_threshold}}]] "
        "$[[--use_worse_than_first_two_pruner ${{inputs.use_worse_than_first_two_pruner}}]] "
        "$[[--study_name ${{inputs.study_name}}]] "
        "--plots ${{outputs.plots}} "
        "--search_db ${{outputs.search_db}} "
        "--feature_names ${{outputs.feature_names}} "
        "--tfidf_scores ${{outputs.tfidf_scores}} "
        "--labels ${{outputs.labels}}"
    ),
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
pushed_component = ml_client.create_or_update(component_command.component)

# Create (register) the component in your workspace
print(
    f"Component {pushed_component.name} with Version {pushed_component.version} is registered"
)
