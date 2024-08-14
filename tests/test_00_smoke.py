import os
from pathlib import Path

import jsonpickle
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

from eppi_text_classification import (
    OptunaHyperparameterOptimisation,
    binary_train_valid_confusion_plotly,
    get_features_and_labels,
    get_tfidf_and_names,
    validation,
)
from eppi_text_classification.opt import delete_optuna_study
from eppi_text_classification.plotly_roc import plotly_roc
from eppi_text_classification.plots import binary_train_valid_confusion_plot
from eppi_text_classification.predict import (
    get_raw_threshold,
    predict_scores,
    raw_threshold_predict,
)
from eppi_text_classification.shap_plotter import (
    BarPlot,
    DecisionPlot,
    DotPlot,
    ShapPlotter,
)


@pytest.fixture(scope="session")
def database_url() -> str:
    if "AML_CloudName" in os.environ:
        print("in")
        return f"sqlite:////mnt/tmp/optuna.db"
    return f"sqlite:///{Path(__file__).parent.parent}/optuna.db"


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


@pytest.fixture(scope="session")
def lgbm_binary_best_params(tfidf_scores, labels, database_url):
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "LGBMClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
        db_url=database_url,
    )
    delete_optuna_study(database_url, "lgbm_binary")
    return optimiser.optimise_hyperparameters(study_name="lgbm_binary")


@pytest.fixture(scope="session")
def xgb_binary_best_params(tfidf_scores, labels, database_url):
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "XGBClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
        db_url=database_url,
    )
    delete_optuna_study(database_url, "xgb_binary")
    return optimiser.optimise_hyperparameters(study_name="xgb_binary")


@pytest.fixture(scope="session")
def svc_binary_best_params(tfidf_scores, labels, database_url):
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "SVC",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
        db_url=database_url,
    )
    delete_optuna_study(database_url, "svc_binary")
    return optimiser.optimise_hyperparameters(study_name="svc_binary")


@pytest.fixture(scope="session")
def randforest_binary_best_params(tfidf_scores, labels, database_url):
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        "RandomForestClassifier",
        n_trials_per_job=1,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=1,
        db_url=database_url,
    )
    delete_optuna_study(database_url, "rf_binary")
    return optimiser.optimise_hyperparameters(study_name="rf_binary")


@pytest.fixture(scope="session")
def Xtrain_Xtest_ytrain_ytest(tfidf_scores, labels):
    return train_test_split(
        tfidf_scores, labels, test_size=0.333, stratify=labels, random_state=8
    )


@pytest.fixture(scope="session")
def Xtrain(Xtrain_Xtest_ytrain_ytest):
    return Xtrain_Xtest_ytrain_ytest[0]


@pytest.fixture(scope="session")
def Xtest(Xtrain_Xtest_ytrain_ytest):
    return Xtrain_Xtest_ytrain_ytest[1]


@pytest.fixture(scope="session")
def ytrain(Xtrain_Xtest_ytrain_ytest):
    return Xtrain_Xtest_ytrain_ytest[2]


@pytest.fixture(scope="session")
def ytest(Xtrain_Xtest_ytrain_ytest):
    return Xtrain_Xtest_ytrain_ytest[3]


def test_optuna_db_path(database_url):
    """Test to ensure the database path is correct."""
    validation.check_valid_database_path(database_url)


def test_delete_optuna_study(database_url):
    """Create a study, delete it, and check it is gone."""
    delete_optuna_study(database_url, "test_study")

    # Create a study and check it exists
    study = optuna.create_study(study_name="test_study", storage=database_url)
    study.optimize(lambda x: x, n_trials=10)
    assert len(study.trials) == 10, "Study did not run 10 trials"
    all_studies = optuna.study.get_all_study_summaries(storage=database_url)
    study_names = [study.study_name for study in all_studies]
    assert "test_study" in study_names, "Study was not created"

    # Delete the study and check it is gone
    delete_optuna_study(database_url, "test_study")
    all_studies = optuna.study.get_all_study_summaries(storage=database_url)
    study_names = [study.study_name for study in all_studies]
    assert "test_study" not in study_names, "Study was not deleted"


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
    study_name, expected_types, best_params, db_storage_url
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
    study = optuna.load_study(study_name=study_name, storage=db_storage_url)

    trial_values = [trial.value for trial in study.trials]
    assert study.best_value == max(trial_values), "Best study was not selected"

    for key, value in study.best_params.items():
        assert key in best_params, f"Key {key} not found in best_params"
        assert (
            best_params[key] == value
        ), f"Value for key {key} does not match: {best_params[key]} != {value}"


def check_binary_optuna_hyperparameter_ranges(
    study_name, expected_ranges, best_params, db_storage_url
):
    study = optuna.load_study(study_name=study_name, storage=db_storage_url)

    for trial in study.trials:
        trial_params = jsonpickle.decode(trial.user_attrs["all_params"], keys=True)
        for param, range_checker in expected_ranges.items():
            assert range_checker(trial_params[param]), f"Param {param} is out of range"


def test_lgbm_binary_optuna_runs(lgbm_binary_best_params):
    """Test to ensure the LGBM hyperparameter optimisation runs."""
    assert lgbm_binary_best_params is not None, "LGBM best params are None"


def test_xgb_binary_optuna_runs(xgb_binary_best_params):
    """Test to ensure the XGB hyperparameter optimisation runs."""
    assert xgb_binary_best_params is not None, "XGB best params are None"


def test_svc_binary_optuna_runs(svc_binary_best_params):
    """Test to ensure the SVC hyperparameter optimisation runs."""
    assert svc_binary_best_params is not None, "SVC best params are None"


def test_randforest_binary_optuna_runs(randforest_binary_best_params):
    """Test to ensure the randforest hyperparameter optimisation runs."""
    assert randforest_binary_best_params is not None, "randforest best params are None"


def test_scale_pos_weights(
    svc_binary_best_params,
    xgb_binary_best_params,
    lgbm_binary_best_params,
    randforest_binary_best_params,
    labels,
):
    true_scale_pos_weight = labels.count(0) / labels.count(1)

    assert (
        svc_binary_best_params["class_weight"] == {0: 1, 1: true_scale_pos_weight}
        or svc_binary_best_params["class_weight"] == "balanced"
    ), "SVC scale_pos_weight is not correct"

    assert (
        xgb_binary_best_params["scale_pos_weight"] == true_scale_pos_weight
    ), "XGBoost scale_pos_weight is not correct"

    assert (
        lgbm_binary_best_params["scale_pos_weight"] == true_scale_pos_weight
    ), "LGBM scale_pos_weight is not correct"

    assert (
        randforest_binary_best_params["class_weight"]
        == {0: 1, 1: true_scale_pos_weight}
        or randforest_binary_best_params["class_weight"] == "balanced"
    ), "Random Forest scale_pos_weight is not correct"


def test_lgbm_binary_optuna_hyperparameter_optimisation(
    lgbm_binary_best_params, database_url
):
    "Test to ensure the LGBM hyperparameters are correct"

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
        "scale_pos_weight": (float, int),
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
        "lgbm_binary", expected_types, lgbm_binary_best_params, database_url
    )

    check_binary_optuna_hyperparameter_ranges(
        "lgbm_binary", expected_ranges, lgbm_binary_best_params, database_url
    )


def test_xgb_binary_optuna_hyperparameter_optimisation(
    xgb_binary_best_params, database_url
):
    "Test to ensure the XGB hyperparameters are correct"
    expected_types = {
        "verbosity": int,
        "objective": str,
        "eval_metric": str,
        "scale_pos_weight": (float, int),
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
        "reg_lambda": lambda x: x >= 0,
        "reg_alpha": lambda x: x >= 0,
        "learning_rate": lambda x: 0 < x <= 1,
        "max_depth": lambda x: x >= 1,
    }

    general_binary_optuna_hyperparameter_checks(
        "xgb_binary", expected_types, xgb_binary_best_params, database_url
    )

    check_binary_optuna_hyperparameter_ranges(
        "xgb_binary", expected_ranges, xgb_binary_best_params, database_url
    )


def test_svc_binary_optuna_hyperparameter_optimisation(
    svc_binary_best_params, database_url
):
    "Test to ensure the SVC hyperparameters are correct"
    expected_types = {
        "class_weight": (str, dict),
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
        "svc_binary", expected_types, svc_binary_best_params, database_url
    )

    check_binary_optuna_hyperparameter_ranges(
        "svc_binary", expected_ranges, svc_binary_best_params, database_url
    )


def test_randforest_binary_optuna_hyperparameter_optimisation(
    randforest_binary_best_params, database_url
):
    "Test to ensure the randforest hyperparameters are correct"
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
        "class_weight": (dict, str),
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
        "rf_binary", expected_types, randforest_binary_best_params, database_url
    )

    check_binary_optuna_hyperparameter_ranges(
        "rf_binary", expected_ranges, randforest_binary_best_params, database_url
    )


@pytest.fixture(scope="session")
def lgbm_binary_optuna_model(lgbm_binary_best_params, Xtrain, ytrain):
    model = LGBMClassifier(**lgbm_binary_best_params)
    model.fit(Xtrain, ytrain)
    return model


def test_lgbm_binary_model_creation(lgbm_binary_optuna_model):
    assert isinstance(lgbm_binary_optuna_model, LGBMClassifier)


@pytest.fixture(scope="session")
def xgb_binary_optuna_model(xgb_binary_best_params, Xtrain, ytrain):
    model = XGBClassifier(**xgb_binary_best_params)
    model.fit(Xtrain, ytrain)
    return model


def test_xgb_binary_model_creation(xgb_binary_optuna_model):
    assert isinstance(xgb_binary_optuna_model, XGBClassifier)


@pytest.fixture(scope="session")
def svc_binary_optuna_model(svc_binary_best_params, Xtrain, ytrain):
    model = SVC(**svc_binary_best_params)
    model.fit(Xtrain, ytrain)
    return model


def test_svc_binary_model_creation(svc_binary_optuna_model):
    assert isinstance(svc_binary_optuna_model, SVC)


@pytest.fixture(scope="session")
def randforest_binary_optuna_model(randforest_binary_best_params, Xtrain, ytrain):
    model = RandomForestClassifier(**randforest_binary_best_params)
    model.fit(Xtrain, ytrain)
    return model


def test_randforest_binary_model_creation(randforest_binary_optuna_model):
    assert isinstance(randforest_binary_optuna_model, RandomForestClassifier)


@pytest.fixture(scope="session")
def all_models(
    lgbm_binary_optuna_model,
    xgb_binary_optuna_model,
    svc_binary_optuna_model,
    randforest_binary_optuna_model,
):
    return (
        lgbm_binary_optuna_model,
        xgb_binary_optuna_model,
        svc_binary_optuna_model,
        randforest_binary_optuna_model,
    )


def test_predict_scores(Xtest, all_models):
    for model in all_models:
        scores = predict_scores(model, Xtest)
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(Xtest)
        assert isinstance(
            scores[0], (np.float64 | np.float32)
        ), f"expected float, got {type(scores[0])}"


def test_plotly_roc(lgbm_binary_optuna_model, Xtest, ytest):
    lgbm_scores = predict_scores(lgbm_binary_optuna_model, Xtest)
    plotly_roc(ytest, lgbm_scores)


def test_get_raw_threshold(all_models, Xtest, ytest):
    for model in all_models:
        threshold = get_raw_threshold(model, Xtest, ytest)
        ypred = predict_scores(model, Xtest)
        assert isinstance(
            threshold, (np.float32 | np.float64)
        ), f"expected float, got {type(threshold)}"
        assert np.min(ypred) <= threshold <= np.max(ypred)


def test_raw_threshold_predict(all_models, Xtest, ytest):
    for model in all_models:
        threshold = get_raw_threshold(model, Xtest, ytest)
        ypred = raw_threshold_predict(model, Xtest, threshold)
        assert isinstance(ypred, np.ndarray)
        assert len(ypred) == len(ytest)
        assert isinstance(ypred[0], np.int_), f"expected int, got {type(ypred[0])}"
        assert set(np.unique(ypred)) == {
            0,
            1,
        }, f"expected [0, 1], got {np.unique(ypred)}"


def test_binary_train_valid_confusion_plot():
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]
    y_train_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    y_test = [1, 0, 0, 0, 0, 0, 0, 0]
    y_test_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    binary_train_valid_confusion_plot(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        positive_label="1",
        negative_label="0",
    )
    plt.close("all")


def test_plotly_binary_train_valid_confusion():
    y_train = [0, 0, 0, 0, 1, 1, 1, 1]
    y_train_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    y_test = [1, 0, 0, 0, 0, 0, 0, 0]
    y_test_pred = [0, 0, 0, 1, 1, 1, 1, 1]
    binary_train_valid_confusion_plotly(
        y_train,
        y_train_pred,
        y_test,
        y_test_pred,
        postive_label="Included",
        negative_label="Excluded",
    )


def test_shap_plotter(all_models, Xtest, ytest, feature_names):
    for model in all_models:
        shap_plotter = ShapPlotter(model, Xtest[:10], feature_names)

        # Test dot plots
        dot_plot = shap_plotter.dot_plot(num_display=10, log_scale=True)
        assert isinstance(dot_plot, DotPlot), f"expected dot plot, got {type(dot_plot)}"
        dot_plot.show()

        dot_plot = shap_plotter.dot_plot(num_display=10, log_scale=False)
        assert isinstance(dot_plot, DotPlot), f"expected dot plot, got {type(dot_plot)}"
        dot_plot.show()

        # Test bar charts
        bar_chart = shap_plotter.bar_chart(num_display=10)
        assert isinstance(
            bar_chart, BarPlot
        ), f"expected bar chart, got {type(bar_chart)}"
        bar_chart.show()
        plt.close("all")

        # Test decision plots
        threshold = get_raw_threshold(model, Xtest, ytest, target_tpr=1)

        decision_plot = shap_plotter.decision_plot(
            threshold=threshold,
            num_display=10,
            log_scale=False,
        )
        assert isinstance(
            decision_plot, DecisionPlot
        ), f"expected decision plot, got {type(decision_plot)}"
        decision_plot.show()

        decision_plot = shap_plotter.decision_plot(
            threshold=threshold,
            num_display=10,
            log_scale=True,
        )
        assert isinstance(
            decision_plot, DecisionPlot
        ), f"expected decision plot, got {type(decision_plot)}"
        decision_plot.show()

        decision_plot = shap_plotter.single_decision_plot(
            threshold=threshold,
            num_display=10,
            log_scale=True,
            index=0,
        )
        assert isinstance(
            decision_plot, DecisionPlot
        ), f"expected decision plot, got {type(decision_plot)}"
        decision_plot.show()

        decision_plot = shap_plotter.single_decision_plot(
            threshold=threshold,
            num_display=10,
            log_scale=False,
            index=0,
        )
        assert isinstance(
            decision_plot, DecisionPlot
        ), f"expected decision plot, got {type(decision_plot)}"
        decision_plot.show()
        plt.close("all")


# def test_raw_threshold(Xtest, ytest):


# Should do some tests to ensure SHAP values add up to the model outputs
# Should do some tests to ensure that the best params are actualy selected by making your
# own instances.
