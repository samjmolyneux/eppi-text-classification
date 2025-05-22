import os
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
    load_dir: str,
) -> None:
    txt_path = os.path.join(load_dir, "model.txt")
    ubj_path = os.path.join(load_dir, "model.ubj")
    joblib_path = os.path.join(load_dir, "model.joblib")

    if os.path.isfile(txt_path):
        return lgb.Booster(model_file=txt_path)

    if os.path.isfile(ubj_path):
        return xgb.Booster(model_file=ubj_path)

    if os.path.isfile(joblib_path):
        return joblib.load(joblib_path)

    msg = f"Could not load model from {load_dir}"
    raise ValueError(msg)


def load_model_from_filepath(
    filepath: str,
) -> None:
    if filepath.endswith(".txt"):
        return lgb.Booster(model_file=filepath)

    if filepath.endswith(".ubj"):
        return xgb.Booster(model_file=filepath)

    if filepath.endswith(".joblib"):
        return joblib.load(filepath)

    msg = f"Could not load model from {filepath}"
    raise ValueError(msg)
