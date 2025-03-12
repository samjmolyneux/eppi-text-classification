from azure.ai.ml import load_component
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

component = load_component(source="./predict_unlabelled_tfidf.yml")

ml_client.components.create_or_update(component)
