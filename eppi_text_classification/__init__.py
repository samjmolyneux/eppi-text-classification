"""Init file for the eppi_text_classification package."""

# from beartype.claw import beartype_this_package

from .multi_process_hparam_search import MultiProcessHparamSearch, delete_optuna_study
from .plots.confusion_plot import (
    binary_train_valid_confusion_plotly,
    binary_train_valid_test_confusion_plotly,
)
from .plots.roc import roc_plot
from .plots.shap_plotter import ShapPlotter
from .predict import get_raw_threshold, predict_scores, raw_threshold_predict
from .save_features_labels import get_features, get_labels, get_tfidf_and_names

# beartype_this_package()

__all__ = [
    "MultiProcessHparamSearch",
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
