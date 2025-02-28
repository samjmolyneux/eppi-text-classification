import json

import jsonpickle
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from load_azure_ml import get_azure_ml_client

ml_client = get_azure_ml_client()

path = "../user_inputs/hyperparameter_search_ranges.json"

hyperparameter_search_ranges = {
    "min_child_samples": {"low": 1, "high": 30, "log": False, "suggest_type": "int"},
    "learning_rate": {"low": 0.1, "high": 0.6, "log": False, "suggest_type": "float"},
    "num_leaves": {"low": 2, "high": 50, "log": False, "suggest_type": "int"},
    "n_estimators": {"low": 100, "high": 500, "log": False, "suggest_type": "int"},
    "min_split_gain": {"low": 1e-6, "high": 10, "log": False, "suggest_type": "float"},
    "min_child_weight": {
        "low": 1e-6,
        "high": 1e-1,
        "log": True,
        "suggest_type": "float",
    },
    "reg_alpha": {"low": 1e-5, "high": 10, "log": True, "suggest_type": "float"},
    "reg_lambda": {"low": 1e-5, "high": 10, "log": True, "suggest_type": "float"},
}

with open(path, "w") as file:
    encoded_ranges = jsonpickle.encode(hyperparameter_search_ranges)
    json.dump(hyperparameter_search_ranges, file)

my_data = Data(
    path=path,
    type=AssetTypes.URI_FILE,
    description="Hyperparameter ranges for search",
    name="hyperparameter_search_ranges",
    version="1.0.0",
)
ml_client.data.create_or_update(my_data)
