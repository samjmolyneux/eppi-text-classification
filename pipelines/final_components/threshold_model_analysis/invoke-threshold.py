from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)
endpoint_name = "threshold-model-analy-adcsd"

inputs = {
    "labelled_tfidf_path": Input(
        type="string",
        default="first_pipeline_output/12/labelled_tfidf.npz",
    ),
    "labels_path": Input(
        type="string",
        default="first_pipeline_output/12/labels.npy",
    ),
    "model_name": Input(
        type="string",
        default="lightgbm",
    ),
    "model_params_path": Input(
        type="string",
        default="first_pipeline_output/12/best_hparams.json",
    ),
    "threshold": Input(
        type="number",
        default=-12.5504,
    ),
    "output_container_path": Input(
        type="string",
        default="first_pipeline_output/12",
    ),
    "nfolds": Input(
        type="integer",
        default=3,
    ),
    "histogram_num_cv_repeats": Input(
        type="integer",
        default=100,
    ),
    "confusion_num_cv_repeats": Input(
        type="integer",
        default=1,
    ),
    "working_container_url": Input(
        type="string",
        default="https://eppidev2985087618.blob.core.windows.net/sams-prod-simulation",
    ),
    "managed_identity_client_id": Input(
        type="string",
        default="df5b7af0-a55a-44d9-9ec7-9cde9abf3051",
    ),
}
job = ml_client.batch_endpoints.invoke(
    endpoint_name=endpoint_name,
    inputs=inputs,
    # deployment_name="default",
    # experiment_name="find_single_model_batch",
)

print("submitted job:", job.name)
# print("studio link  :", job.services["Studio"].endpoint)
