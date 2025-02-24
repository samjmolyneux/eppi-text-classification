from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

data_path = "../data/raw/debunking_review.tsv"

debunking_data = Data(
    name="debunking_review_data",
    path=data_path,
    type=AssetTypes.URI_FILE,
    description="Dataset for testing sams text classification pipeline",
    version="1.0.0",
)

debunking_data = ml_client.data.create_or_update(debunking_data)
print(
    f"Dataset with name {debunking_data.name} was registered to workspace, the dataset version is {debunking_data.version}"
)
