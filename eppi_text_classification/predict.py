"""Universal functions for making model predictions."""

import numpy as np
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .validation import InvalidModelError


def get_raw_threshold(
    model: LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC,
    X: NDArray[np.float64],
    y: NDArray[np.int_],
    target_tpr: float = 1,
) -> np.float32 | np.float64:
    """
    Get the model prediction threshold required to achieve the target TPR.

    Given a binary classifier, using a raw prediction score, find the minimun threshold
    that is required to achieve a target true positive that is larger than or equal to
    target_tpr.

    Parameters
    ----------
    model : LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC
        A trained model.

    X : np.ndarray
        Data to classifiy in the shape (samples, features).

    y : np.ndarray
        True labels for the data.

    target_tpr : float[0, 1], optional
        Target true positive rate. By default 1.0.

    Returns
    -------
    float
        The model prediction threshold required to achieve the target TPR.

    """
    y_test_scores = predict_scores(model, X)
    _, tpr, thresholds = roc_curve(y, y_test_scores)
    idx = np.searchsorted(tpr, target_tpr)
    return thresholds[idx]


def raw_threshold_predict(
    model: LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC,
    X: NDArray[np.float64],
    threshold: float,
) -> NDArray[np.int_]:
    """
    Universal function to predict binary labels using a raw score threshold.

    Parameters
    ----------
    model : LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC
        Classifier to make prediction with.

    X : np.ndarray
        Samples to make predictions on.

    threshold : float
        Threshold to use for binary classification.

    Returns
    -------
    np.ndarray[int]
        Binary predictions for X.

    """
    y_pred_scores = predict_scores(model, X)
    return (y_pred_scores >= threshold).astype(int)


def predict_scores(
    model: LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC,
    X: NDArray[np.float64],
) -> NDArray[np.float64] | NDArray[np.float32]:
    """
    Make raw score predictions for a binary classifier.

    The function works as a wrapper for the predict method of the model.
    The function exclusively predicts using raw scores for binary classification,
    meaning the prediction before no transformation or thresholding. This ensures
    that these scores can be used for precisely calculating SHAP values. For models
    where raw score cannot be predicted, predict_proba can often function as a raw
    score.

    Parameters
    ----------
    model : LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC
        Binary classifier.

    X : np.ndarray
        Data to make predictions on.

    Returns
    -------
    np.ndarray[float]
        Raw prediction scores for binary classification.

    """
    if isinstance(model, LGBMClassifier):
        return model.predict(X, raw_score=True)
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True)
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, SVC):
        return model.decision_function(X)

    raise InvalidModelError(model)
