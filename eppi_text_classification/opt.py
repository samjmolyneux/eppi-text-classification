"""Methods for optimsing hyperparameters for models."""

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import jsonpickle
import numpy as np
import optuna
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import validation

# Considerations: Will the database work correctly in deployment?
# Considerations: Need a way to handle the interupts
# Considerations: The cache size needs setting for the SVC

# TO DO: Add a way for mutliclass
# TO DO: Add a way to automatically fill the class weights for each objective function
# TO DO: Check the defaults are all good for the params
# TO DO: Fix all the params

# URGENT TO DO: MAKE SURE ALL THE MODELS A SINGLE CORE


@dataclass
class LGBMParams:
    """Dataclass for LightGBM hyperparameters."""

    verbosity: int = -1
    boosting_type: str = "gbdt"
    max_depth: int = 7
    min_child_samples: int = 20
    learning_rate: float = 0.33
    num_leaves: int = 31
    n_estimators: int = 500
    subsample_for_bin: int = 20000
    subsample: float = 1.0
    objective: str = "binary"
    scale_pos_weight: int = 27
    min_split_gain: float = 0
    min_child_weight: float = 1e-3
    reg_alpha: float = 0
    reg_lambda: float = 0


@dataclass
class XGBParams:
    """Dataclass for XGBoost hyperparameters."""

    verbosity: int = 0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    scale_pos_weight: int = 1
    n_estimators: int = 1000
    colsample_bytree: float = 1.0
    n_jobs: int = 1
    reg_lambda: float = 0.0
    reg_alpha: float = 0.0
    learning_rate: float = 0.01
    max_depth: int = 1


@dataclass
class SVCParams:
    """Dataclass for SVC hyperparameters."""

    class_weight: str = "balanced"
    cache_size: int = 1000
    probability: bool = False
    C: float = 1.0
    kernel: str = "rbf"
    shrinking: bool = True
    tol: float = 1e-3
    gamma: str | float = "scale"


@dataclass
class RandForestParams:
    """Dataclass for RandomForest hyperparameters."""

    verbose: int = 0
    n_estimators: int = 100
    criterion: str = "gini"
    n_jobs: int = 1
    max_depth: int | None = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_features: str = "sqrt"
    max_leaf_nodes: int | None = None
    min_impurity_decrease: float = 0.0
    bootstrap: bool = True
    class_weight: dict[int, int] = field(default_factory=lambda: {1: 27, 0: 1})
    ccp_alpha: float = 0.0
    max_samples: int | None = None
    monotonic_cst: int | None = None


mname_to_mclass = {
    "SVC": SVC,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

model_to_selector = {
    "SVC": "select_svc_hyperparameters",
    "LGBMClassifier": "select_lgbm_hyperparameters",
    "RandomForestClassifier": "select_rand_forest_hyperparameters",
    "XGBClassifier": "select_xgb_hyperparameters",
}


class OptunaHyperparameterOptimisation:
    """An engine for optimsing hyperparameters for a using optuna."""

    def __init__(
        self,
        tfidf_scores: NDArray[np.float64],
        labels: Sequence[int],
        model: str,
        n_trials_per_job: int = 200,
        n_jobs: int = -1,
        nfolds: int = 3,
        num_cv_repeats: int = 3,
        db_url: str | None = None,
    ) -> None:
        """
        Build a new hyperparameter optimisation engine.

        Parameters
        ----------
        tfidf_scores : np.ndarray
            Tfidf scores for the text data, shape (n_samples, n_features).

        labels : Sequence[int]
            Labels corresponding to the text data, shape (n_samples,).

        model : SVC | LGBMClassifier | RandomForestClassifier | XGBClassifier
            Classification model to optimise.

        n_trials_per_job : int, optional
            Number of optimisation trials each processor will perform, by default 200.

        n_jobs : int, optional
            Number of parallel processes to use. Setting n_jobs=-1 will use all
            available processes. By default -1.

        nfolds : int, optional
            Number of folds to use when performing cross-validation for evalutating
            model performance. By default 3.

        num_cv_repeats : int, optional
            Number of times to repeat and average the cross validation scores.
            Small datasets may be prone to validation set overfitting.
            By repeating the cross validation, the selected hyperparameters
            become more generalisable.
            A different random seed is used for each stratified fold.
            By default 3.

        db_url : str, optional
            URL to the database to store the hyperparameter search history.
            If None, a database will be created in the current working directory.
            !!!!!!!DO NOT USE NONE IF RUNNING ON AZURE ML STUDIO!!!!!!!!!!!!
            By default None.

        """
        validation.check_valid_model(model)

        self.tfidf_scores = tfidf_scores
        self.labels = labels
        self.n_trials_per_job = n_trials_per_job
        self.nfolds = nfolds
        self.num_cv_repeats = num_cv_repeats

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs

        self.model_class = mname_to_mclass[model]
        self.select_hyperparameters = getattr(self, model_to_selector[model])

        self.interupt = False

        if db_url is None:
            root_path = Path(Path(__file__).resolve()).parent.parent
            self.db_storage_url = f"sqlite:///{root_path}/optuna.db"
        else:
            self.db_storage_url = db_url

        validation.check_valid_database_path(self.db_storage_url)

        self.positive_class_weight = labels.count(0) / labels.count(1)

    def optimise_on_single(self, n_trials: int, study_name: str) -> None:
        """
        Run the hyperparameter search for a single process.

        For a hyperaparameter search, given by study_name,
        controls the hyperparmeter optimisation search for a single process.
        This method will not start a new study, but will add an additional process
        to search the hyperparmeter space of an existing study.

        Parameters
        ----------
        n_trials : int
            Number of trials to run on the given process.

        study_name : str
            Name of the study. This is what tracks our hyperparameter search.

        """
        study = optuna.load_study(study_name=study_name, storage=self.db_storage_url)
        study.optimize(self.objective_func, n_trials=n_trials, n_jobs=1)

    def optimise_hyperparameters(
        self,
        study_name: str,
    ) -> dict[str, Any]:
        """
        Initiate the hyperparameter search.

        Parameters
        ----------
        study_name : str
            A name to assign to a hyperparmeter search. Allows for stopping
            and continuing the search at a later time.

        Returns
        -------
        dict
            Model hyperparameters that resulted in best cross-validation performance
            during the search. Key: hyperparameter name, value: hyperparameter value.

        """
        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_storage_url,
            direction="maximize",
            load_if_exists=True,
        )
        try:
            # TO DO: try and remove the square brackets
            Parallel(n_jobs=-1)(
                [
                    delayed(self.optimise_on_single)(self.n_trials_per_job, study_name)
                    for _ in range(self.n_jobs)
                ]
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

        best_trial = study.best_trial
        best_params = best_trial.user_attrs["all_params"]
        best_params = jsonpickle.decode(best_params, keys=True)

        return best_params

    def objective_func(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for the hyperparameter search.

        Uses the search history to determine the next set of hyperparameters to try.
        Then calculates the cross validation ROC-AUC score of the model with the
        given hyperparameters.

        Parameters
        ----------
        trial : optuna.trial.Trial
            An individual trial in the hyperparameter search.

        Returns
        -------
        float
            The cross validation ROC-AUC score of the model with the given
            hyperparameters of the trial.

        """
        params = self.select_hyperparameters(trial)
        serialized_params = jsonpickle.encode(asdict(params), keys=True)
        trial.set_user_attr("all_params", serialized_params)
        clf = self.model_class(**asdict(params))

        # Calculate the cross validation score
        scores = []
        for i in range(self.num_cv_repeats):
            kf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=i)
            scores.extend(
                cross_val_score(
                    clf, self.tfidf_scores, self.labels, cv=kf, scoring="roc_auc"
                )
            )
        return np.mean(scores)

    def select_lgbm_hyperparameters(self, trial: optuna.trial.Trial) -> LGBMParams:
        """
        Select LightGBM hyperparameters for a given iteration in the search.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial in the hyperparameter search.

        Returns
        -------
        LGBMParams
            The selected hyperparameters for the LightGBM model.

        """
        return LGBMParams(
            verbosity=-1,
            boosting_type="gbdt",
            max_depth=trial.suggest_int("max_depth", 1, 15),
            min_child_samples=trial.suggest_int("min_child_samples", 1, 30),
            learning_rate=trial.suggest_float("learning_rate", 0.1, 0.6),
            num_leaves=trial.suggest_int("num_leaves", 2, 50),
            n_estimators=trial.suggest_int("n_estimators", 100, 3000, log=True),
            subsample_for_bin=20000,
            subsample=1.0,
            objective="binary",
            scale_pos_weight=self.positive_class_weight,
            min_split_gain=trial.suggest_float("min_split_gain", 1e-6, 10, log=True),
            min_child_weight=trial.suggest_float(
                "min_child_weight", 1e-6, 1e-1, log=True
            ),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-5, 10, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-5, 10, log=True),
        )

    def select_xgb_hyperparameters(self, trial: optuna.trial.Trial) -> XGBParams:
        """
        Select XGBoost hyperparameters for a given iteration in the search.

        Parameters
        ----------
        trial : optuna.trial.Trial
            An individual trial in the hyperparameter search

        Returns
        -------
        XGBParams
            The selected hyperparameters for the XGBoost model.

        """
        # TO DO: sort params out to right format
        return XGBParams(
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=self.positive_class_weight,
            n_estimators=1000,
            colsample_bytree=1,
            n_jobs=1,
            reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 100, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 100, log=True),
            learning_rate=trial.suggest_float("learning_rate", 1e-2, 1, log=True),
            max_depth=trial.suggest_int("max_depth", 1, 5),
        )

    def select_svc_hyperparameters(self, trial: optuna.trial.Trial) -> SVCParams:
        """
        Select SVC hyperparameters for a given iteration in the search.

        Parameters
        ----------
        trial : optuna.trial.Trial
            An individual trial in the hyperparameter search

        Returns
        -------
        SVCParams
            The selected hyperparameters for the SVC model.

        """
        # TO DO: Sort these params out
        return SVCParams(
            class_weight={1: self.positive_class_weight, 0: 1},
            cache_size=1000,
            probability=False,
            C=trial.suggest_float("C", 1e-3, 10000, log=True),
            kernel="rbf",
            shrinking=True,
            tol=1e-8,
        )

    def select_rand_forest_hyperparameters(
        self, trial: optuna.trial.Trial
    ) -> RandForestParams:
        """
        Select RandomForest hyperparameters for a given iteration in the search.

        Parameters
        ----------
        trial : optuna.trial.Trial
            An individual trial in the hyperparameter search

        Returns
        -------
        RandForestParams
            The selected hyperparameters for the Random Forest model.

        """
        return RandForestParams(
            verbose=0,
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            criterion="gini",
            n_jobs=1,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            class_weight={1: self.positive_class_weight, 0: 1},
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
        )

    def delete_optuna_study(self, study_name: str) -> None:
        """
        Delete an optuna study from the database at self.db_storage_url.

        Parameters
        ----------
        study_name : str
            Name of the study to delete.

        """
        delete_optuna_study(self.db_storage_url, study_name)


def delete_optuna_study(db_url: str, study_name: str) -> None:
    """
    If it exists, delete an optuna study from the database.

    Parameters
    ----------
    db_url : str
        URL to the database where the study is stored.

    study_name : str
        Name of the study to delete.

    """
    all_studies = optuna.study.get_all_study_summaries(storage=db_url)
    study_names = [study.study_name for study in all_studies]
    if study_name in study_names:
        optuna.delete_study(study_name=study_name, storage=db_url)
