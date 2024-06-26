from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

model_list = [
    "SVC",
    "LGBMClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "XGBClassifier",
]


class InvalidModelError(Exception):
    def __init__(self, model):
        super().__init__(f"Model must be one of {model_list}, but got {model}.")


def check_valid_model(model):
    if model not in model_list:
        raise InvalidModelError(model)
