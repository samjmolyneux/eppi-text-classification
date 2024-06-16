import os
from pathlib import Path

import numpy as np
import optuna
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Considerations: Will the database work correctly in deployment?
# Considerations: Need a way to handle the interupts
# Considerations: The cache size needs setting for the SVC

# TO DO: Add a way for mutliclass
# TO DO: Add a way to automatically fill the class weights for each objective function
# TO DO: Check the defaults are all good for the params
# TO DO: Fix all the params

# URGENT TO DO: MAKE SURE ALL THE MODELS A SINGLE CORE


model_list = [
    "SVC",
    "LGBMClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "XGBClassifier",
]

model_to_opt_func = {
    "SVC": "SVC_objective",
    "LGBMClassifier": "LGBM_objective",
    "RandomForestClassifier": "RandomForest_objective",
    "LogisticRegression": "LogisticRegression_objective",
    "XGBClassifier": "XGB_objective",
}


class OptunaHyperparameterOptimisation:
    def __init__(
        self,
        tfidf_scores,
        labels,
        model,
        n_trials_per_job=200,
        n_jobs=-1,
        nfolds=3,
        num_cv_repeats=3,
    ):
        assert model in model_list, f"Model must be one of {model_list}"

        self.tfidf_scores = tfidf_scores
        self.labels = labels
        self.n_trials_per_job = n_trials_per_job
        self.nfolds = nfolds
        self.num_cv_repeats = num_cv_repeats

        if n_jobs == -1:
            self.n_jobs = os.cpu_count()
        else:
            self.n_jobs = n_jobs

        self.objective = getattr(self, model_to_opt_func[model])

        self.interupt = False

        root_path = Path(Path(__file__).resolve()).parent.parent
        self.db_storage_url = f"sqlite:///{root_path}/optuna.db"

    def optimise_on_single(self, n_trials, study_name):
        study = optuna.load_study(study_name=study_name, storage=self.db_storage_url)
        study.optimize(self.objective, n_trials=n_trials)

    def optimise_hyperparameters(
        self,
        study_name,
    ):
        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_storage_url,
            direction="maximize",
            load_if_exists=True,
        )

        # TO DO: try and remove the square brackets
        Parallel(n_jobs=-1)(
            [
                delayed(self.optimise_on_single)(self.n_trials_per_job, study_name)
                for _ in range(self.n_jobs)
            ]
        )

        best_trial = study.best_trial
        best_params = best_trial.user_attrs["all_params"]

        return best_params

    def get_cv_performance(self, clf):
        scores = []
        for i in range(self.num_cv_repeats):
            kf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=i)
            scores.extend(
                cross_val_score(
                    clf, self.tfidf_scores, self.labels, cv=kf, scoring="roc_auc"
                )
            )
        average = np.mean(scores)
        return average

    def LGBM_objective(self, trial):
        if self.interupt:
            return float("nan")

        try:
            param = {
                "verbosity": -1,
                "boosting_type": "gbdt",
                "max_depth": 7,
                "min_child_samples": 20,  # Must be more than this many samples in a leaf
                "learning_rate": 0.33,
                "num_leaves": 31,
                "n_estimators": 500,
                "subsample_for_bin": 20000,
                "subsample": 1.0,
                "objective": "binary",
                "scale_pos_weight": 27,
                "min_split_gain": 0,
                "min_child_weight": 1e-3,
                "reg_alpha": 0,  # L1
                "reg_lambda": 0,  # L2
            }

            param["n_estimators"] = trial.suggest_int(
                "n_estimators", 100, 3000, log=True
            )
            param["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-5, 10, log=True)
            param["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-5, 10, log=True)
            param["min_child_samples"] = trial.suggest_int("min_child_samples", 1, 30)
            param["max_depth"] = trial.suggest_int("max_depth", 1, 15)
            param["learning_rate"] = trial.suggest_float("learning_rate", 0.1, 0.6)
            param["num_leaves"] = trial.suggest_int("num_leaves", 2, 50)
            param["min_child_weight"] = trial.suggest_float(
                "min_child_weight", 1e-6, 1e-1, log=True
            )
            param["min_split_gain"] = trial.suggest_float(
                "min_split_gain", 1e-6, 10, log=True
            )

            trial.set_user_attr("all_params", param)

            clf = LGBMClassifier(**param)

            performance = self.get_cv_performance(clf)

            return performance

        except KeyboardInterrupt:
            self.interupt = True
            return float("nan")

    def XGB_objective(self, trial):
        if self.interupt:
            return float("nan")
        # TO DO: sort params out to right format
        param = {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": 27,
            "n_estimators": 1000,
            "colsample_bytree": 1,
            "n_jobs": 1,
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 100, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 100, log=True),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-2, 100, log=True
            ),  # Same as eta
            # "n_estimators": trial.suggest_int("n_estimators", 100000, 150000),
            # "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
            # "gamma": trial.suggest_float("gamma", 1e-12, 1e-5, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
        }
        try:
            trial.set_user_attr("all_params", param)

            clf = XGBClassifier(**param)

            performance = self.get_cv_performance(clf)

            return performance

        except KeyboardInterrupt:
            self.interupt = True
            return float("nan")

    def SVC_objective(self, trial):
        if self.interupt:
            return float("nan")

        try:
            # TO DO: Sort these params out
            param = {
                "class_weight": "balanced",
                "cache_size": 1000,
                "probability": False,
                "C": trial.suggest_float("C", 1e-3, 10000, log=True),
                "kernel": "rbf",
                "shrinking": False,
                "tol": 1e-8,
                # TO DO: FIND A BETTER WAY OF SETTING THIS
                # "gamma": trial.suggest_float("gamma", 1e-9, 1e-2, log=True),
            }

            param["C"] = trial.suggest_float("C", 1e-3, 10000, log=True)

            if param["kernel"] == "sigmoid":
                param["coef0"] = trial.suggest_float("coef0", 1e-12, 100, log=True)

            trial.set_user_attr("all_params", param)

            clf = SVC(**param)

            performance = self.get_cv_performance(clf)

            return performance

        except KeyboardInterrupt:
            self.interupt = True
            return float("nan")

    def RandomForest_objective(self, trial):
        if self.interupt:
            return float("nan")

        try:
            param = {
                "verbose": 0,
                "n_estimators": 100,
                "criterion": "gini",
                "n_jobs": 1,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0.0,
                "max_features": "sqrt",
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0.0,
                "bootstrap": True,
                "class_weight": 27,
                "ccp_alpha": 0.0,
                "max_samples": None,
                "monotonic_cst": None,
            }

            param["n_estimators"] = trial.suggest_int("n_estimators", 100, 1000)

            trial.set_user_attr("all_params", param)

            clf = RandomForestClassifier(**param)

            performance = self.get_cv_performance(clf)

            return performance

        except KeyboardInterrupt:
            self.interupt = True
            return float("nan")
