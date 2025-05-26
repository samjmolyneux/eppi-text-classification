from azure.ai.ml import load_component

from pipelines import get_azure_ml_client, get_registry_client

# ml_client = get_azure_ml_client()
ml_client = get_registry_client()

component = load_component(source="./classify_unlabelled_tfidf.yml")

ml_client.components.create_or_update(component)
