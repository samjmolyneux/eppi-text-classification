from sklearn.metrics import roc_curve
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# TO DO: Just double check that the roc_curve and roc are correct.
def get_threshold(model, X, y, target_tpr=1):
    y_test_probabilities = predict_probabilities(model, X)
    _, tpr, thresholds = roc_curve(y, y_test_probabilities)
    idx = np.searchsorted(tpr, target_tpr)
    return thresholds[idx]


def predict_probabilities(model, X):
    if isinstance(model, LGBMClassifier | XGBClassifier | RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, SVC):
        return model.decision_function(X)

    return "Model not supported"


def threshold_predict(model, X, threshold):
    if isinstance(model, LGBMClassifier | XGBClassifier | RandomForestClassifier):
        y_pred_prob = model.predict_proba(X)[:, 1]
    if isinstance(model, SVC):
        y_pred_prob = model.decision_function(X)

    return (y_pred_prob >= threshold).astype(int)
