import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential


def get_azure_ml_client(resource_group_name=None, workspace_name=None):
    # Get credentials for login
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        print("We are doing interactive")
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
    resource_group_name = (
        resource_group_name
        if resource_group_name
        else os.getenv("AZURE_RESOURCE_GROUP")
    )
    workspace_name = workspace_name if workspace_name else os.getenv("AZURE_WORKSPACE")
    return MLClient(
        credential=credential,
        subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID"),
        resource_group_name=resource_group_name,
        workspace_name=workspace_name,
    )


def get_registry_client():
    # Get credentials for login
    try:
        credential = DefaultAzureCredential()
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        # Fall back to InteractiveBrowserCredential in case DefaultAzureCredential not work
        print("We are doing interactive")
        credential = InteractiveBrowserCredential()

    from dotenv import load_dotenv

    load_dotenv()

    # Get a handle to the workspace
    return MLClient(
        credential=credential,
        registry_name=os.getenv("AZURE_REGISTRY_NAME"),
    )
