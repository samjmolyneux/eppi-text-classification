from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)
endpoint_name = "classify-unlabelled-tfidf-aljvd"

inputs = {
    "unlabelled_tfidf_path": Input(
        type="string",
        default="first_pipeline_output/12/unlabelled_tfidf.npz",
    ),
    "threshold": Input(
        type="number",
        default=13,
    ),
    "trained_model_dir": Input(
        type="string",
        default="first_pipeline_output/12/trained_model",
    ),
    "output_container_path": Input(
        type="string",
        default="first_pipeline_output/12",
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
    deployment_name="other",
    inputs=inputs,
)

print("Batch job submitted :", job.name)
