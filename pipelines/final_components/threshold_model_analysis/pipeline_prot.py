from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

prev_job = ml_client.jobs.get("ee31b20b-e5b6-4982-a490-736aba7397dd")
print(prev_job.description)
print(prev_job.properties)
print(prev_job.outputs)
