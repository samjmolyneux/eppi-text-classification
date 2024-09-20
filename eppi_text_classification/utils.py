"""Utility functions for azure ml."""

import json
import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import jsonpickle
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import load_npz

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from scipy.sparse import csr_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier


def load_np_array_at_directory(directory_path: str, allow_pickle=False) -> NDArray[Any]:
    """Load numpy array from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    return np.load(file_path, allow_pickle=allow_pickle)


def load_json_at_directory(directory_path: str) -> dict[str, Any]:
    """Load json from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    with file_path.open() as file:
        dict_from_json = jsonpickle.decode(json.load(file))
    return dict_from_json


def load_value_from_json_at_directory(directory_path: str) -> dict[str, Any]:
    """Load value json from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    with file_path.open() as file:
        value = json.load(file)
    return value


def load_csr_at_directory(directory_path: str) -> "csr_matrix":
    """Load csr matrix from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    return load_npz(file_path)


def load_joblib_model_at_directory(
    directory_path: str,
) -> "LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC":
    """Load joblib model from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    return joblib.load(file_path)


def load_pickle_object_at_directory(directory_path: str) -> Any:
    """Load pickle object from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    with file_path.open("rb") as file:
        return pickle.load(file)
