from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from eppi_text_classification import (
    OptunaHyperparameterOptimisation,
    get_features_and_labels,
    get_tfidf_and_names,
)
from eppi_text_classification.utils import delete_optuna_study


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Fixture to load the data from a TSV file."""
    df_path = "data/raw/debunking_review.tsv"
    df = pd.read_csv(df_path, sep="\t")
    return df


@pytest.fixture(scope="session")
def features_and_labels(raw_df: pd.DataFrame) -> tuple[list[str], list[int]]:
    word_features, labels = get_features_and_labels(raw_df)
    return word_features, labels


@pytest.fixture(scope="session")
def tfidf_and_names(
    features_and_labels: tuple[list[str], list[int]],
) -> tuple[pd.DataFrame, list[str]]:
    features, labels = features_and_labels
    tfidf_scores, feature_names = get_tfidf_and_names(features)
    return tfidf_scores, feature_names


@pytest.fixture(scope="session")
def tfidf_scores(tfidf_and_names):
    return tfidf_and_names[0]


@pytest.fixture(scope="session")
def feature_names(tfidf_and_names):
    return tfidf_and_names[1]


@pytest.fixture(scope="session")
def labels(features_and_labels):
    return features_and_labels[1]


def test_load_data(raw_df):
    """Test to ensure data is loaded properly."""
    df = raw_df
    assert not df.empty, "DataFrame is empty"
    assert "title" in df.columns, "Title column is missing"
    assert "abstract" in df.columns, "Abstract column is missing"
    assert "included" in df.columns, "Included column is missing"


def test_get_features_and_labels(features_and_labels):
    """Test to ensure features and labels are returned properly."""
    features, labels = features_and_labels
    assert len(features) == len(labels), "Features and labels are not the same length"
    assert isinstance(features, list), "Features are not a list"
    assert isinstance(labels, list), "Labels are not a list"
    assert isinstance(features[0], str), "Features are not strings"
    assert isinstance(labels[0], int), "Labels are not integers"


def test_get_tfidf_and_names(tfidf_and_names):
    """Test to ensure tfidf scores and feature names are returned properly."""
    tfidf_scores, feature_names = tfidf_and_names
    assert isinstance(tfidf_scores, np.ndarray), "Tfidf scores are not a np.ndarray"
    assert isinstance(tfidf_scores[0][0], np.float64), "Tfidf scores are not floats"
    assert isinstance(feature_names, np.ndarray), "Feature names are not a list"
    assert isinstance(feature_names[0], str), "Feature names are not strings"
    assert len(tfidf_scores[0]) == len(
        feature_names
    ), "Tfidf scores and feature have different numbers of features"


def general_binary_optuna_hyperparameter_checks(
    study_name, expected_types, best_params
):
    assert isinstance(best_params, dict), "Best parameters are not a dictionary"

    # Check expected types cover all params
    for param in best_params.keys():
        assert param in expected_types, f"Key {param} not found in expected_types"

    # Check all expected types are in best_params
    for param in expected_types.keys():
        assert param in best_params, f"Key {param} not found in best_params"

    # Check each param is of the expected type
    for param, expected_type in expected_types.items():
        assert isinstance(
            best_params[param], expected_type
        ), f"{param} is not of type {expected_type}"

    # Now just check its the best one.
    root_path = Path(Path(__file__).resolve()).parent.parent
    db_storage_url = f"sqlite:///{root_path}/optuna.db"
    study = optuna.load_study(study_name=study_name, storage=db_storage_url)

    trial_values = [trial.value for trial in study.trials]
    assert study.best_value == max(trial_values), "Best study was not selected"

    for key, value in study.best_params.items():
        assert key in best_params, f"Key {key} not found in best_params"
        assert (
            best_params[key] == value
        ), f"Value for key {key} does not match: {best_params[key]} != {value}"


def check_binary_optuna_hyperparameter_ranges(expected_ranges, best_params):
    for param, range_checker in expected_ranges.items():
        assert range_checker(best_params[param]), f"Param {param} is out of range"


def test_lgbm_binary_optuna_hyperparameter_optimisation(
    tfidf_scores, labels, feature_names
):
    """Test to ensure the LGBM hyperparameter optimisation runs."""
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "LGBMClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
    )
    delete_optuna_study("lgbm_binary")
    best_params = optimiser.optimise_hyperparameters(study_name="lgbm_binary")

    expected_types = {
        "verbosity": int,
        "boosting_type": str,
        "max_depth": int,
        "min_child_samples": int,
        "learning_rate": (float, int),
        "num_leaves": int,
        "n_estimators": int,
        "subsample_for_bin": int,
        "subsample": (float, int),
        "objective": str,
        "scale_pos_weight": int,
        "min_split_gain": (float, int),
        "min_child_weight": (float, int),
        "reg_alpha": (float, int),
        "reg_lambda": (float, int),
    }

    expected_ranges = {
        "verbosity": lambda x: x in [-1, 0, 1, 2, 3],
        "boosting_type": lambda x: x in ["gbdt", "dart", "goss", "rf"],
        "max_depth": lambda x: x == -1 or x >= 1,
        "min_child_samples": lambda x: x >= 1,
        "learning_rate": lambda x: x >= 0,
        "num_leaves": lambda x: x >= 1,
        "n_estimators": lambda x: x >= 1,
        "subsample_for_bin": lambda x: x >= 1,
        "subsample": lambda x: 0 < x <= 1,
        "objective": lambda x: x in ["binary"],
        "scale_pos_weight": lambda x: x >= 0,
        "min_split_gain": lambda x: x >= 0,
        "min_child_weight": lambda x: x >= 0,
        "reg_alpha": lambda x: x >= 0,
        "reg_lambda": lambda x: x >= 0,
    }

    general_binary_optuna_hyperparameter_checks(
        "lgbm_binary", expected_types, best_params
    )

    check_binary_optuna_hyperparameter_ranges(expected_ranges, best_params)


def test_xgb_binary_optuna_hyperparameter_optimisation(
    tfidf_scores, labels, feature_names
):
    """Test to ensure the XGBoost hyperparameter optimisation runs."""
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "XGBClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
    )
    delete_optuna_study("xgb_binary")
    best_params = optimiser.optimise_hyperparameters(study_name="xgb_binary")

    expected_types = {
        "verbosity": int,
        "objective": str,
        "eval_metric": str,
        "scale_pos_weight": int,
        "n_estimators": int,
        "colsample_bytree": (float, int),
        "n_jobs": int,
        "reg_lambda": (float, int),
        "reg_alpha": (float, int),
        "learning_rate": (float, int),
        "max_depth": int,
    }

    expected_ranges = {
        "verbosity": lambda x: x in [-1, 0, 1, 2, 3],
        "objective": lambda x: x
        in ["binary:logistic", "binary:logitraw", "binary:hinge"],
        "eval_metric": lambda x: x in ["logloss"],
        "scale_pos_weight": lambda x: x >= 0,
        "n_estimators": lambda x: x >= 1,
        "colsample_bytree": lambda x: 0 < x <= 1,
        "n_jobs": lambda x: x == -1 or x >= 1,
        "reg_lambda": lambda x: 0 <= x <= 1,
        "reg_alpha": lambda x: 0 <= x <= 1,
        "learning_rate": lambda x: 0 < x <= 1,
        "max_depth": lambda x: x >= 1,
    }

    general_binary_optuna_hyperparameter_checks(
        "xgb_binary", expected_types, best_params
    )

    check_binary_optuna_hyperparameter_ranges(expected_ranges, best_params)


def test_svc_binary_optuna_hyperparameter_optimisation(
    tfidf_scores, labels, feature_names
):
    """Test to ensure the XGBoost hyperparameter optimisation runs."""
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "SVC",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
    )
    delete_optuna_study("svc_binary")
    best_params = optimiser.optimise_hyperparameters(study_name="svc_binary")

    expected_types = {
        "class_weight": (str, dict[int, float]),
        "cache_size": int,
        "probability": bool,
        "C": (float, int),
        "kernel": str,
        "shrinking": bool,
        "tol": (float, int),
        "gamma": (str, float),
    }

    expected_ranges = {
        "class_weight": lambda x: x in ["balanced"]
        if isinstance(x, str)
        else set(x.keys()) == {0, 1} and x[0] >= 0 and x[1] >= 0,
        "cache_size": lambda x: x >= 0,
        "probability": lambda x: x in [True, False],
        "C": lambda x: x >= 0,
        "kernel": lambda x: x in ["linear", "poly", "rbf", "sigmoid", "precomputed"],
        "shrinking": lambda x: x in [True, False],
        "tol": lambda x: x >= 0,
        "gamma": lambda x: x in ["scale", "auto"] or x >= 0,
    }

    general_binary_optuna_hyperparameter_checks(
        "svc_binary", expected_types, best_params
    )

    check_binary_optuna_hyperparameter_ranges(expected_ranges, best_params)


def test_randforest_binary_optuna_hyperparameter_optimisation(
    tfidf_scores, labels, feature_names
):
    """Test to ensure the XGBoost hyperparameter optimisation runs."""
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "RandomForestClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
    )
    delete_optuna_study("rf_binary")
    best_params = optimiser.optimise_hyperparameters(study_name="rf_binary")

    expected_types = {
        "verbose": int,
        "n_estimators": int,
        "criterion": str,
        "n_jobs": int,
        "max_depth": (int, type(None)),
        "min_samples_split": (int, float),
        "min_samples_leaf": int,
        "min_weight_fraction_leaf": (float, int),
        "max_features": (str, int, float),
        "max_leaf_nodes": (int, type(None)),
        "min_impurity_decrease": (float, int),
        "bootstrap": bool,
        "class_weight": dict,
        "ccp_alpha": (float, int),
        "max_samples": (int, type(None)),
        "monotonic_cst": type(None),
    }

    expected_ranges = {
        "verbose": lambda x: x in [-1, 0, 1, 2, 3],
        "n_estimators": lambda x: x >= 1,
        "criterion": lambda x: x in ["gini", "entropy", "logloss"],
        "n_jobs": lambda x: x == -1 or x >= 1,
        "max_depth": lambda x: x is None or x >= 1,
        "min_samples_split": lambda x: x >= 2 if isinstance(x, int) else 0 <= x <= 1,
        "min_samples_leaf": lambda x: x >= 1 if isinstance(x, int) else 0 <= x <= 1,
        "min_weight_fraction_leaf": lambda x: 0 <= x <= 1,
        "max_features": lambda x: (isinstance(x, str) and x in ["auto", "sqrt", "log2"])
        or (isinstance(x, int) and x >= 1)
        or (isinstance(x, float) and 0 < x <= 1),
        "max_leaf_nodes": lambda x: x is None or x >= 1,
        "min_impurity_decrease": lambda x: x >= 0,
        "bootstrap": lambda x: x in [True, False],
        "class_weight": lambda x: set(x.keys()) == {0, 1} and x[0] >= 0 and x[1] >= 0,
        "ccp_alpha": lambda x: x >= 0,
        "max_samples": lambda x: x is None
        or (isinstance(x, int) and x >= 1)
        or (isinstance(x, float) and 0 < x <= 1),
        "monotonic_cst": lambda x: x is None,
    }

    general_binary_optuna_hyperparameter_checks(
        "rf_binary", expected_types, best_params
    )


# Want to check that each param in best_params is in the expected_type

# Should also test that the hyperparmeters are also in the correct range.
# Should do some tests to ensure SHAP values add up to the model outputs
# Should do some tests to ensure that the best params are actualy selected by making your
# own instances.
