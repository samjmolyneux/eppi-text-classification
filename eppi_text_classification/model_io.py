from typing import TYPE_CHECKING

import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from .validation import InvalidModelError


def save_model_to_dir(
    model: lgb.basic.Booster | RandomForestClassifier | xgb.core.Booster | SVC,
    save_dir: str,
) -> None:
    print(f"Saving model from to {save_dir}")
    if isinstance(model, lgb.Booster):
        model.save_model(f"{save_dir}/model.txt")
    elif isinstance(model, xgb.Booster):
        model.save_model(f"{save_dir}/model.ubj")
    elif isinstance(model, RandomForestClassifier | SVC):
        joblib.dump(model, f"{save_dir}/model.joblib")
    else:
        raise InvalidModelError(model)


def load_model_from_dir(
    model_name: str,
    load_dir: str,
) -> None:
    print(f"Loading model from {load_dir}")
    if model_name == "lightgbm":
        lgb.Booster(model_file=f"{load_dir}/model.txt")
    if model_name == "xgboost":
        xgb.Booster(model_file=f"{load_dir}/model.ubj")
    if model_name == "RandomForestClassifier":
        joblib.load(f"{load_dir}/model.joblib")
    if model_name == "SVC":
        joblib.load(f"{load_dir}/model.joblib")

    msg = f"Model {model_name} not recognised"
    raise ValueError(msg)
