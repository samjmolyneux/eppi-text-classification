import os

from azure.ai.ml import Input, Output, command
from load_azure_ml import get_azure_ml_client, get_current_package_env

ml_client = get_azure_ml_client()

pipeline_job_env = get_current_package_env(ml_client)

splice_data = "./../primitive_components/splice_data"

splice_data_component = command(
    name="splice_csr_data_for_classifier_workbench",
    display_name="Splice data for csr matrix for classifier workbench",
    description=("Splice data for csr matrix for classifier workbench"),
    inputs={
        "data": Input(type="uri_folder"),
        "num_rows": Input(type="integer"),
    },
    outputs={
        "spliced_data": Output(type="uri_folder", mode="rw_mount"),
    },
    # The source folder of the component
    code=splice_data,
    command="""python splice_data.py \
            --data ${{inputs.data}} \
            --num_rows ${{inputs.num_rows}} \
            --spliced_data ${{outputs.spliced_data}} \
            """,
    environment=f"{pipeline_job_env.name}:{pipeline_job_env.version}",
)

# Now we register the component to the workspace
splice_data_component = ml_client.create_or_update(
    splice_data_component.component, version="prim_1.0"
)

# Create (register) the component in your workspace
print(
    f"Component {splice_data_component.name} with Version {splice_data_component.version} is registered"
)
