from sklearn.metrics import roc_curve
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# TO DO: Just double check that the roc_curve and roc are correct.
# def get_proba_threshold(model, X, y, target_tpr=1):
#     y_test_probabilities = predict_probabilities(model, X)
#     _, tpr, thresholds = roc_curve(y, y_test_probabilities)
#     idx = np.searchsorted(tpr, target_tpr)
#     return thresholds[idx]


# def predict_probabilities(model, X):
#     if isinstance(model, LGBMClassifier | XGBClassifier | RandomForestClassifier):
#         return model.predict_proba(X)[:, 1]
#     if isinstance(model, SVC):
#         return model.decision_function(X)

#     return "Model not supported"


# def proba_threshold_predict(model, X, threshold):
#     y_pred_prob = predict_probabilities(model, X)
#     return (y_pred_prob >= threshold).astype(int)


def get_raw_threshold(model, X, y, target_tpr=1):
    y_test_scores = predict_scores(model, X)
    _, tpr, thresholds = roc_curve(y, y_test_scores)
    idx = np.searchsorted(tpr, target_tpr)
    return thresholds[idx]


def raw_threshold_predict(model, X, threshold):
    y_pred_scores = predict_scores(model, X)
    return (y_pred_scores >= threshold).astype(int)


def predict_scores(model, X):
    if isinstance(model, LGBMClassifier):
        return model.predict(X, raw_score=True)
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True)
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, SVC):
        return model.decision_function(X)

    return "Model not supported"
