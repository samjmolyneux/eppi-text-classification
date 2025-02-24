import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_mlflow_env

ml_client = get_azure_ml_client()

display_image_env = get_mlflow_env(ml_client)

view_html_image = "./components/view_html_image"

view_html_image_component = command(
    name="view_html_image",
    display_name="Display image from html file in logs",
    description=("Display image from html file in logs"),
    inputs={
        "image": Input(type="uri_folder"),
    },
    # The source folder of the component
    code=view_html_image,
    command="""python view_html_image.py \
            --image ${{inputs.image}} \
            """,
    environment=f"{display_image_env.name}:{display_image_env.version}",
)
