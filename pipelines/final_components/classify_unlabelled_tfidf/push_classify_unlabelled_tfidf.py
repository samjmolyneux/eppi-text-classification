from azure.ai.ml import load_component

from pipelines import get_azure_ml_client, get_registry_client

# ml_client = get_azure_ml_client(
#     workspace_name="EPPI_PROD_ML",
#     resource_group_name="eppi_ml_prod_rg",
# )

ml_client = get_registry_client()

# upload component to registry
component = load_component(source="./classify_unlabelled_tfidf.yml")
ml_client.components.create_or_update(component)

# upload pipeline component to registry
component = load_component(source="./classify_pipeline.yml")
ml_client.components.create_or_update(component)
