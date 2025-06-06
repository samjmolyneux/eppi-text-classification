from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)
inputs = {
    "labelled_data_path": Input(
        type="string",
        default="raw-data/debunking_review.tsv",
    ),
    "unlabelled_data_path": Input(
        type="string",
        default="raw-data/debunking_review.tsv",
    ),
    "model_name": Input(
        type="string",
        default="lightgbm",
    ),
    "output_container_path": Input(
        type="string",
        default="first_pipeline_output/12",
    ),
    "title_header": Input(
        type="string",
        default="title",
    ),
    "abstract_header": Input(
        type="string",
        default="abstract",
    ),
    "label_header": Input(
        type="string",
        default="included",
    ),
    "positive_class_value": Input(
        type="string",
        default="1",
    ),
    "nfolds": Input(
        type="integer",
        default=3,
    ),
    "num_cv_repeats": Input(
        type="integer",
        default=1,
    ),
    "timeout": Input(
        type="integer",
        default=86400,
    ),
    "use_early_terminator": Input(
        type="boolean",
        default=False,
    ),
    "use_worse_than_first_two_pruner": Input(
        type="boolean",
        default=False,
    ),
    "shap_num_display": Input(
        type="integer",
        default=20,
    ),
    "working_container_url": Input(
        type="string",
        default="https://eppidev2985087618.blob.core.windows.net/sams-prod-simulation",
    ),
    "managed_identity_client_id": Input(
        type="string",
        default="df5b7af0-a55a-44d9-9ec7-9cde9abf3051",
    ),
    # ----------- optional params (uncomment to use) --------
    # "hparam_search_ranges_path":       Input(type="string",  default="https://<storage>/ranges.yaml"),
    "max_n_search_iterations": Input(
        type="integer",
        default=100,
    ),
    # "max_stagnation_iterations":       Input(type="integer", default=25),
    # "wilcoxon_trial_pruner_threshold": Input(type="number",  default=0.05),
}

job = ml_client.batch_endpoints.invoke(
    endpoint_name="find-model-classifier-wbench",
    inputs=inputs,
    experiment_name="find_single_model_batch",
)

print("submitted job:", job.name)
# print("studio link  :", job.services["Studio"].endpoint)
