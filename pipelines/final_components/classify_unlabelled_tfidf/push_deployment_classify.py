from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

component_name = "classify_unlabelled_tfidf_for_classifier_workbench_pipeline"
compute_name = "classifier-workbench-20c-140gb"
endpoint_name = "classify-unlabelled-tfidf-aljvd"

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)

registry_client = get_registry_client()

component = registry_client.components.get(component_name)
print(component.id)
compute_cluster = ml_client.compute.get(compute_name)


deployment = PipelineComponentBatchDeployment(
    name="default",
    endpoint_name=endpoint_name,
    component=component.id,
    settings={
        "default_compute": compute_cluster.name,
        "continue_on_step_failure": False,
    },
)

ml_client.batch_deployments.begin_create_or_update(deployment)
