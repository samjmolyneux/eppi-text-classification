import copy
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .utils import SuppressStderr

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def train(
    model_name: str,
    params: dict[str, Any],
    X: "csr_matrix",
    y: NDArray[np.int64],
    n_jobs: int = -1,
) -> NDArray[np.float64] | NDArray[np.float32]:
    train_params = copy.deepcopy(params)
    train_params["n_jobs"] = n_jobs

    if model_name == "lightgbm":
        with SuppressStderr(
            [
                "No further splits with positive gain, best gain: -inf",
                "Stopped training because there are no more leaves that meet the split requirements",
            ]
        ):
            num_boost_round = train_params.pop("num_iterations")
            dtrain = lgb.Dataset(X, label=y, free_raw_data=True)
            model = lgb.train(train_params, dtrain, num_boost_round=num_boost_round)
        return model

    if model_name == "xgboost":
        num_boost_round = train_params.pop("n_estimators")
        # enable categorical is always false
        enable_categorical = train_params.pop("enable_categorical")
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=enable_categorical)
        # xgboost.train ignores params it cannot use
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
        )
        # gblinear uses a different shap explainer, so we save booster info
        model.set_attr(boosting_type=train_params["booster"])
        return model

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**train_params)
        model.fit(X, y)
        return model

    if model_name == "SVC":
        # SVC does not support n_jobs
        train_params.pop("n_jobs", None)
        model = SVC(**train_params)
        model.fit(X, y)
        return model

    print(f"model is {model}")
    msg = "Model not recognised."
    raise ValueError(msg)
