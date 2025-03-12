from azure.ai.ml import Input, load_component
from azure.ai.ml.dsl import pipeline
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

cluster_name = "sams-16core-DS5-v2"
print(ml_client.compute.get(cluster_name))

# find_single_model = ml_client.components.get(
#     "find_single_model_for_classifier_workbench"
# )

find_single_model = load_component(
    source="./final_components/find_single_model/find_single_model.yml"
)

input_data = ml_client.data.get(name="debunking_review_data", version="1.0.0")

hyperparameter_search_ranges = Input(
    path="./user_inputs/uri_folders/hyperparameter_search_ranges/", type="uri_file"
)


@pipeline(default_compute=cluster_name, name="find_single_model_pipeline")
def find_single_model_pipeline(
    data,
    hyperparameter_search_ranges,
    title_header,
    abstract_header,
    label_header,
    positive_class_value,
    model_name,
    max_n_search_iterations,
    nfolds,
    num_cv_repeats,
    timeout,
    use_early_terminator,
    max_stagnation_iterations,
    wilcoxon_trial_pruner_threshold,
    use_worse_than_first_two_pruner,
    shap_num_display,
):
    find_run = find_single_model(
        data=data,
        title_header=title_header,
        abstract_header=abstract_header,
        label_header=label_header,
        positive_class_value=positive_class_value,
        model_name=model_name,
        hyperparameter_search_ranges=hyperparameter_search_ranges,
        max_n_search_iterations=max_n_search_iterations,
        nfolds=nfolds,
        num_cv_repeats=num_cv_repeats,
        timeout=timeout,
        use_early_terminator=use_early_terminator,
        max_stagnation_iterations=max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=use_worse_than_first_two_pruner,
        shap_num_display=shap_num_display,
    )

    return find_run.outputs


first_pipeline = find_single_model_pipeline(
    data=input_data,
    hyperparameter_search_ranges=hyperparameter_search_ranges,
    title_header="title",
    abstract_header="abstract",
    label_header="included",
    positive_class_value=1,
    model_name="lightgbm",
    max_n_search_iterations=100,
    nfolds=3,
    num_cv_repeats=1,
    timeout=None,
    use_early_terminator=False,
    max_stagnation_iterations=None,
    wilcoxon_trial_pruner_threshold=None,
    use_worse_than_first_two_pruner=False,
    shap_num_display=20,
)

first_pipeline_job = ml_client.jobs.create_or_update(
    first_pipeline, experiment_name="find_model_sdk"
)
