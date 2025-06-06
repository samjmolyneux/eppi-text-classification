from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import BatchEndpoint, PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)

endpoint = BatchEndpoint(
    name="classify-unlabelled-tfidf-aljvd",
)

ml_client.batch_endpoints.begin_create_or_update(endpoint).result()
