"""Universal functions for making model predictions."""

from typing import TYPE_CHECKING

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

from .validation import InvalidModelError
from .train import train

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix


def get_raw_threshold(
    model: lgb.basic.Booster | RandomForestClassifier | xgb.core.Booster | SVC,
    X: "csr_matrix",
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

    X : np.ndarray[float]
        Data to classifiy in the shape (samples, features).

    y : np.ndarray[int]
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
    model: lgb.basic.Booster | RandomForestClassifier | xgb.core.Booster | SVC,
    X: "csr_matrix",
    threshold: float,
) -> NDArray[np.int_]:
    """
    Universal function to predict binary labels using a raw score threshold.

    Parameters
    ----------
    model : LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC
        Classifier to make prediction with.

    X : np.ndarray[float]
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
    model: lgb.basic.Booster | RandomForestClassifier | xgb.core.Booster | SVC,
    X: "csr_matrix",
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
    if isinstance(model, lgb.Booster):
        return model.predict(X, raw_score=True)
    if isinstance(model, xgb.Booster):
        dtrain = xgb.DMatrix(X)
        return model.predict(dtrain, output_margin=True)
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, SVC):
        return model.decision_function(X)

    raise InvalidModelError(model)


def predict_cv_metrics(tfidf_scores, labels, model_name, model_params, nfolds, num_cv_repeats, threshold=None):
    auc_scores = []
    recall_scores = []
    specifcity_scores = []

    for i in range(num_cv_repeats):
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)
        for _, (train_idx, val_idx) in enumerate(
            kf.split(tfidf_scores, labels)
        ):
            X_train = tfidf_scores[train_idx]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx]
            y_val = labels[val_idx]

            clf = train(model_name, model_params, X_train, y_train)

            y_val_scores = predict_scores(clf, X_val)

            auc = roc_auc_score(y_val, y_val_scores)
            auc_scores.append(auc)

            if threshold is not None:
                y_val_pred = raw_threshold_predict(clf, X_val, threshold)
                recall = recall_score(y_val, y_val_pred)
                recall_scores.append(recall)

                specifcity = recall_score(y_val, y_val_pred, pos_label=0)
                specifcity_scores.append(specifcity)

    if threshold is None:
        return auc_scores
    else: 
        return auc_scores, recall_scores, specifcity_scores   


def predict_cv_scores(tfidf_scores, labels, model_name, model_params, nfolds, num_cv_repeats):
    fold_raw_scores = []

    for i in range(num_cv_repeats):
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)
        for _, (train_idx, val_idx) in enumerate(
            kf.split(tfidf_scores, labels)
        ):
            X_train = tfidf_scores[train_idx]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx]

            clf = train(model_name, model_params, X_train, y_train)

            y_val_scores = predict_scores(clf, X_val)
            fold_raw_scores.append(y_val_scores)

    return fold_raw_scores

def predict_cv_metrics_per_model(tfidf_scores, labels, model_names, model_params_list, nfolds, num_cv_repeats, thresholds=None):
    auc_scores_per_model = []
    recall_scores_per_model = []
    specificity_scores_per_model = []

    assert len(model_names) == len(model_params_list), "model_list and model_param_list must have the same length"
    if thresholds is not None:
        assert len(model_names) == len(thresholds), "model_list and thresholds must have the same length"

    for i in range(len(model_names)):

        model_name = model_names[i]
        model_params = model_params_list[i]
        if thresholds is not None:
            threshold = thresholds[i]

        auc_scores = []
        recall_scores = []
        specifcity_scores = []
        for i in range(num_cv_repeats):
            kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)

            for _, (train_idx, val_idx) in enumerate(
                kf.split(tfidf_scores, labels)
            ):
                X_train = tfidf_scores[train_idx]
                X_val = tfidf_scores[val_idx]

                y_train = labels[train_idx]
                y_val = labels[val_idx]

                clf = train(model_name, model_params, X_train, y_train)

                y_val_scores = predict_scores(clf, X_val)

                auc = roc_auc_score(y_val, y_val_scores)
                auc_scores.append(auc)

                if thresholds is not None:
                    y_val_pred = raw_threshold_predict(clf, X_val, threshold)
                    recall = recall_score(y_val, y_val_pred)
                    recall_scores.append(recall)

                    specifcity = recall_score(y_val, y_val_pred, pos_label=0)
                    specifcity_scores.append(specifcity)

        auc_scores_per_model.append(auc_scores)
        recall_scores_per_model.append(recall_scores)
        specificity_scores_per_model.append(specifcity_scores)

    if thresholds is None:
        return auc_scores_per_model
    else: 
        return auc_scores_per_model, recall_scores_per_model, specificity_scores_per_model  
    