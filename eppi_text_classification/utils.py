"""Utility functions for azure ml."""

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import jsonpickle
import numpy as np
from scipy.sparse import load_npz

if TYPE_CHECKING:
    from lightgbm import LGBMClassifier
    from scipy.sparse import csr_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier


def load_np_array_at_directory(directory_path: str) -> np.ndarray:
    """Load numpy array from directory with single file."""
    file_list = os.listdir(directory_path)
    return np.load(file_list[0])


def load_json_at_directory(directory_path: str) -> dict:
    """Load json from directory with single file."""
    file_list = os.listdir(directory_path)
    with Path(file_list[0]).open() as file:
        dict_from_json = jsonpickle.decode(json.load(file))
    return dict_from_json


def load_csr_at_directory(directory_path: str) -> "csr_matrix":
    """Load csr matrix from directory with single file."""
    file_list = os.listdir(directory_path)
    return load_npz(file_list[0])


def load_joblib_model_at_directory(
    directory_path: str,
) -> "LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC":
    """Load joblib model from directory with single file."""
    file_list = os.listdir(directory_path)
    return joblib.load(file_list[0])
