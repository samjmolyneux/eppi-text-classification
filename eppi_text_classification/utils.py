"""Utility functions for azure ml."""

import json
import os
import pickle
import sys
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


def load_np_array_at_directory(
    directory_path: str, allow_pickle: bool = False
) -> NDArray[Any]:
    """Load numpy array from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    return np.load(file_path, allow_pickle=allow_pickle)


def load_json_at_directory(
    directory_path: str, unpickle: bool = False
) -> dict[str, Any]:
    """Load json from directory with single file."""
    file_path = Path(directory_path) / os.listdir(directory_path)[0]
    with file_path.open() as file:
        if unpickle:
            return jsonpickle.decode(json.load(file))

        return json.load(file)


def load_json_at_fname(
    fname: str,
    unpickle: bool = False,
) -> dict[str, Any]:
    """Load json from directory with single file."""
    with open(fname) as file:
        if unpickle:
            return jsonpickle.decode(json.load(file))

        return json.load(file)


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


def parse_multiple_types(value: str) -> int | float | str:
    """
    Parse a string into an int, float, or str in that hierarchical order.

    Parameters
    ----------
    value : str
        The value to parse.

    Returns
    -------
    int | float | str
        int if can be cast, else if it can be casted to float, float, else str.

    """
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def str2bool(string: str) -> bool:
    if string.lower() == "true":
        return True
    if string.lower() == "false":
        return False

    raise ValueError(f"Cannot convert {string} to boolean.")


class SuppressStderr:
    def __init__(self, messages):
        self.messages = messages
        self.original_stdout = sys.stdout

    def __enter__(self):
        # Redirect stderr to this context manager
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore the original stderr
        sys.stdout = self.original_stdout

    def write(self, output: str):
        # Suppress messages containing the keyword
        if (
            all(message not in output for message in self.messages)
            and not output.isspace()
        ):
            self.original_stdout.write(output)

    def flush(self):
        pass  # To match `sys.stderr` behavior
