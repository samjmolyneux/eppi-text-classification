import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


def get_azure_ml_client():
    # Get credentials for login
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        credential = InteractiveBrowserCredential()

    if (
        not os.getenv("AZURE_SUBSCRIPTION_ID")
        or not os.getenv("AZURE_RESOURCE_GROUP")
        or not os.getenv("AZURE_WORKSPACE")
    ):
        # Import and load dotenv only if environment variables are not already set
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError as exc:
            msg = (
                "dotenv is not installed, and environment variables are missing. "
                "Please install 'python-dotenv' or set the environment variables."
            )
            raise RuntimeError(msg) from exc

    # Get a handle to the workspace
    return MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=os.getenv("AZURE_RESOURCE_GROUP"),
        workspace_name=os.getenv("AZURE_WORKSPACE"),
    )


def get_current_package_env(ml_client: MLClient):
    # Retrieve an existing environment from the workspace
    env_name = "aml-eppi-text-classification"
    env_version = "0.1.3"  # Specify the version of the environment
    return ml_client.environments.get(name=env_name, version=env_version)


def get_mlflow_env(ml_client: MLClient):
    # Retrieve an existing environment from the workspace
    env_name = "display-image-env"
    env_version = "0.1.0"  # Specify the version of the environment
    return ml_client.environments.get(name=env_name, version=env_version)

def get_temp_env(ml_client: MLClient):
    # Retrieve an existing environment from the workspace
    env_name = ""
    env_version = "0.1.0"  # Specify the version of the environment
    return ml_client.environments.get(name=env_name, version=env_version)
