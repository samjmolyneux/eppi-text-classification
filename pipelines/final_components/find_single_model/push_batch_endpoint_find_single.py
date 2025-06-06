from azure.ai.ml import Input
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import PipelineComponentBatchDeployment

from pipelines import get_azure_ml_client, get_registry_client

ml_client = get_azure_ml_client(
    workspace_name="EPPI_PROD_ML",
    resource_group_name="eppi_ml_prod_rg",
)

registry_client = get_registry_client()

# find_model_component = registry_client.components.get(
#     "find_single_model_for_classifier_workbench_pipeline", version="1"
# )
find_model_component = registry_client.components.get(
    "find_single_model_for_classifier_workbench_pipeline", version="1"
)
print(find_model_component.id)
compute_cluster = ml_client.compute.get("classifier-workbench-20c-140gb")


deployment = PipelineComponentBatchDeployment(
    name="workspace",
    endpoint_name="find-model-classifier-wbench",
    component=find_model_component.id,
    settings={
        "default_compute": compute_cluster.name,
        "continue_on_step_failure": False,
    },
)

ml_client.batch_deployments.begin_create_or_update(deployment)


# # So try deploying directly with the component
# # Look at the pipeliecomponent batch deployment and then look at the optiosn on api reference and fill them in
