"""Init file for the eppi_text_classification package."""

# from beartype.claw import beartype_this_package

from .opt import OptunaHyperparameterOptimisation, delete_optuna_study
from .plots.confusion_plot import (
    binary_train_valid_confusion_plotly,
    binary_train_valid_test_confusion_plotly,
)
from .plots.roc import roc_plot
from .predict import get_raw_threshold, predict_scores, raw_threshold_predict
from .save_features_labels import get_features, get_labels, get_tfidf_and_names
from .shap_plotter import ShapPlotter

# beartype_this_package()

__all__ = [
    "OptunaHyperparameterOptimisation",
    "ShapPlotter",
    "binary_train_valid_confusion_plotly",
    "binary_train_valid_test_confusion_plotly",
    "delete_optuna_study",
    "get_features",
    "get_labels",
    "get_raw_threshold",
    "get_tfidf_and_names",
    "predict_scores",
    "raw_threshold_predict",
    "roc_plot",
]
