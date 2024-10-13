"""Methods for optimsing hyperparameters for models."""

import multiprocessing
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonpickle
import numpy as np
import optuna

# from optuna._imports import _LazyImport
import scipy.stats as ss
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from numpy.typing import NDArray
from optuna.pruners import WilcoxonPruner
from optuna.study import MaxTrialsCallback
from optuna.terminator import (
    BestValueStagnationEvaluator,
    CrossValidationErrorEvaluator,
    RegretBoundEvaluator,
    StaticErrorEvaluator,
    Terminator,
    TerminatorCallback,
    report_cross_validation_scores,
)
from optuna.trial import TrialState
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import validation

# ss = _LazyImport("scipy.stats")

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

# Considerations: Will the database work correctly in deployment?
# Considerations: Need a way to handle the interupts
# Considerations: The cache size needs setting for the SVC
# Considerations: Should not use SVC for large datasets

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
    scale_pos_weight: int | float = 27
    min_split_gain: float = 0
    min_child_weight: float = 1e-3
    reg_alpha: float = 0
    reg_lambda: float = 0
    n_jobs: int = 1


@dataclass
class XGBParams:
    """Dataclass for XGBoost hyperparameters."""

    verbosity: int = 0
    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    scale_pos_weight: int | float = 1
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

    class_weight: str | dict[int, float | int] = "balanced"
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
    class_weight: dict[int, int | float] = field(default_factory=lambda: {1: 27, 0: 1})
    ccp_alpha: float = 0.0
    max_samples: int | None = None
    monotonic_cst: int | None = None


# LLIMIT DEFAULT MAX N_ESTIMSTORS TO ABOIUT 1000

default_xgb_hyperparameter_ranges = {
    "n_estimators": {"low": 100, "high": 1000, "log": False},
    "reg_lambda": {"low": 1e-4, "high": 100, "log": True},
    "reg_alpha": {"low": 1e-4, "high": 100, "log": True},
    "learning_rate": {"low": 1e-2, "high": 1, "log": True},
    "max_depth": {"low": 1, "high": 5, "log": False},
}

default_lgbm_hyperparameter_ranges = {
    "max_depth": {"low": 1, "high": 15, "log": False},
    "min_child_samples": {"low": 1, "high": 30, "log": False},
    "learning_rate": {"low": 0.1, "high": 0.6, "log": False},
    "num_leaves": {"low": 2, "high": 50, "log": False},
    "n_estimators": {"low": 100, "high": 1000, "log": False},
    "min_split_gain": {"low": 1e-6, "high": 10, "log": True},
    "min_child_weight": {"low": 1e-6, "high": 1e-1, "log": True},
    "reg_alpha": {"low": 1e-5, "high": 10, "log": True},
    "reg_lambda": {"low": 1e-5, "high": 10, "log": True},
}

default_svc_hyperparameter_ranges = {
    "C": {"low": 1e-3, "high": 100000, "log": True},
    # HOW to do or categoriical? for if we want scale?
    # "gamma": {"low": 1e-7, "high": 1000, "log": True},
}

default_rand_forest_hyperparameter_ranges = {
    "n_estimators": {"low": 100, "high": 1000, "log": False},
}

model_name_to_model_class = {
    "SVC": SVC,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}

model_name_to_selector = {
    "SVC": "select_svc_hyperparameters",
    "LGBMClassifier": "select_lgbm_hyperparameters",
    "RandomForestClassifier": "select_rand_forest_hyperparameters",
    "XGBClassifier": "select_xgb_hyperparameters",
}

model_name_to_hyperparameter_ranges = {
    "SVC": default_svc_hyperparameter_ranges,
    "LGBMClassifier": default_lgbm_hyperparameter_ranges,
    "RandomForestClassifier": default_rand_forest_hyperparameter_ranges,
    "XGBClassifier": default_xgb_hyperparameter_ranges,
}


class OptunaHyperparameterOptimisation:
    """An engine for optimsing hyperparameters for a using optuna."""

    def __init__(
        self,
        tfidf_scores: "csr_matrix",
        labels: NDArray[np.int64],
        model_name: str,
        max_n_search_iterations: int | None = None,
        n_jobs: int = -1,
        nfolds: int = 3,
        num_cv_repeats: int = 3,
        db_url: str | None = None,
        user_selected_hyperparameter_ranges: dict[str, dict] | None = None,
        timeout: float | None = None,
        use_early_terminator: bool = False,
        max_stagnation_iterations: int | None = None,
        wilcoxon_trial_pruner_threshold: float | None = None,
        use_worse_than_first_two_pruner: bool = False,
    ) -> None:
        """
        Build a new hyperparameter optimisation engine.

        This object must be called with a main guard.

        Parameters
        ----------
        tfidf_scores : csr_matrix
            Tfidf scores for the text data, shape (n_samples, n_features).

        labels : Sequence[int]
            Labels corresponding to the text data, shape (n_samples,).

        model_name : "SVC" | "LGBMClassifier" | "RandomForestClassifier"
        | "XGBClassifier"
            Classification model to optimise.

        n_jobs : int, optional
            Number of parallel processes to use. Setting n_jobs=-1 will use all
            available processes. By default -1.

        nfolds : int, optional
            Number of folds to use when performing cross-validation for evalutating
            model performance. Must be larger than 1. By default 3.

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

        user_selected_hyperparameter_ranges : dict, optional
            User selected hyperparameter ranges for search.
            Should be provided in the same format as the default ranges.
            If None, default ranges will be used.

        timeout : float, optional]
            Stop the hyperparameter search after the given number of
            seconds. When None, the search will continue until all trials
            are complete. By default None.


        """
        validation.check_valid_model(model_name)

        assert nfolds > 1, "nfolds must be greater than 1."

        self.tfidf_scores = tfidf_scores
        self.labels = labels
        self.nfolds = nfolds
        self.num_cv_repeats = num_cv_repeats
        self.timeout = timeout
        self.use_early_terminator = use_early_terminator
        self.max_stagnation_iterations = max_stagnation_iterations
        self.max_n_search_iterations = max_n_search_iterations
        self.wilcoxon_trial_pruner_threshold = wilcoxon_trial_pruner_threshold
        self.use_worse_than_first_two_pruner = use_worse_than_first_two_pruner

        # Bool to track if we need to use a pruner
        self.use_pruner = (
            self.wilcoxon_trial_pruner_threshold is not None
            or self.use_worse_than_first_two_pruner
        )

        # We are multiprocessing, so must set up a manager to share when to stop
        manager = multiprocessing.Manager()
        self.stopping_event = manager.Event()

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        print(f"Number of processes: {self.n_jobs}")

        self.model_class = model_name_to_model_class[model_name]
        self.select_hyperparameters = getattr(self, model_name_to_selector[model_name])

        self.setup_database(db_url)

        self.final_hyperparameter_search_ranges = (
            self.define_hyperparameter_search_ranges(
                user_selected_hyperparameter_ranges, model_name
            )
        )

        self.positive_class_weight = np.count_nonzero(labels == 0) / np.count_nonzero(
            labels == 1
        )
        print(f"Positive class weight: {self.positive_class_weight}")

    def setup_database(self, db_url: str | None) -> None:
        """
        Set up the database for the hyperparameter search.

        Parameters
        ----------
        db_url :
            URL to the database to store the hyperparameter search history.

        """
        if db_url is None:
            root_path = Path(Path(__file__).resolve()).parent.parent
            self.db_storage_url = f"sqlite:///{root_path}/optuna.db"
        else:
            self.db_storage_url = db_url

        validation.check_valid_database_url(self.db_storage_url)

        # If a database does not exist, it will be created by optuna.

        print(self.db_storage_url)

    def define_hyperparameter_search_ranges(
        self,
        user_selected_ranges: dict[str, dict] | None,
        model_name: str,
    ) -> dict[str, dict]:
        """
        Define the hyperparameter search ranges for the model.

        By default uses the default ranges for the model. If user selected ranges are
        given, replaces defaults with the given ones.

        Parameters
        ----------
        user_selected_ranges : dict, optional
            User selected hyperparameter ranges. If None, default ranges will be used.

        model_name : str
            Name of the model to optimise.

        Returns
        -------
        dict
            Hyperparameter ranges for the model.

        """
        default_ranges = model_name_to_hyperparameter_ranges[model_name]

        if user_selected_ranges is None:
            return default_ranges

        final_ranges = {}
        for key, default_value in default_ranges.items():
            final_ranges[key] = user_selected_ranges.get(key, default_value)
        print(final_ranges)
        return final_ranges

    def optimise_on_single(self, study_name: str) -> None:
        """
        Run the hyperparameter search for a single process.

        For a hyperaparameter search, given by study_name,
        controls the hyperparmeter optimisation search for a single process.
        This method will not start a new study, but will add an additional process
        to search the hyperparmeter space of an existing study.

        Parameters
        ----------
        study_name : str
            Name of the study. This is what tracks our hyperparameter search.

        """
        study = optuna.load_study(study_name=study_name, storage=self.db_storage_url)

        callbacks = self.create_search_callbacks()

        study.optimize(
            self.objective_func,
            n_jobs=1,
            timeout=self.timeout,
            callbacks=callbacks,
        )

        # One process has finished, set the stopping event to stop all other processes
        # Another process could break the stopping condition, leading to
        # a search on fewer processes.
        self.stopping_event.set()

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
        if self.use_pruner:
            # A pruner must be able to share the best scores between processes
            self.create_best_cv_scores_shared_memory()

        study = optuna.create_study(
            study_name=study_name,
            storage=self.db_storage_url,
            direction="maximize",
            load_if_exists=True,
        )
        try:
            # TO DO: try and remove the square brackets
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(self.optimise_on_single)(study_name)
                    for _ in range(self.n_jobs)
                ]
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

        best_trial = study.best_trial
        best_params = best_trial.user_attrs["all_params"]
        best_params = jsonpickle.decode(best_params, keys=True)

        if self.use_pruner:
            # Once the search is complete, we must clean up the shared memory
            self.delete_best_cv_scores_shared_memory()

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

            for fold_idx, (train_idx, val_idx) in enumerate(
                kf.split(self.tfidf_scores, self.labels)
            ):
                X_train = self.tfidf_scores[train_idx]
                X_val = self.tfidf_scores[val_idx]

                y_train = self.labels[train_idx]
                y_val = self.labels[val_idx]

                clf.fit(X_train, y_train)

                y_val_pred = _predict_scores(clf, X_val)

                auc = roc_auc_score(y_val, y_val_pred)
                scores.append(auc)

                # Prune if need to
                if self.use_pruner:
                    should_prune = self.should_we_prune(trial, scores)
                    if should_prune:
                        print(f"Pruned trial with scores: {scores}")
                        return np.mean(scores)

        # Early terminator uses variance of reported scores to calculate error
        if self.use_early_terminator:
            report_cross_validation_scores(trial, scores)

        print(f"Finished trial with scores: {scores}")

        # Update the shared memory with the best scores
        if self.use_pruner:
            self.update_shared_memory_best_cv_scores(scores)

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
        # Default ranges for LGBMParams
        # "max_depth": {"low": 1, "high": 15, "log": False},  # noqa: ERA001
        # "min_child_samples": {"low": 1, "high": 30, "log": False},  # noqa: ERA001
        # "learning_rate": {"low": 0.1, "high": 0.6, "log": False},  # noqa: ERA001
        # "num_leaves": {"low": 2, "high": 50, "log": False},  # noqa: ERA001
        # "n_estimators": {"low": 100, "high": 3000, "log": True},  # noqa: ERA001
        # "min_split_gain": {"low": 1e-6, "high": 10, "log": True},  # noqa: ERA001
        # "min_child_weight": {"low": 1e-6, "high": 1e-1, "log": True},  # noqa: ERA001
        # "reg_alpha": {"low": 1e-5, "high": 10, "log": True},  # noqa: ERA001
        # "reg_lambda": {"low": 1e-5, "high": 10, "log": True},  # noqa: ERA001

        return LGBMParams(
            verbosity=-1,
            boosting_type="gbdt",
            subsample_for_bin=20000,
            subsample=1.0,
            objective="binary",
            scale_pos_weight=self.positive_class_weight,
            n_jobs=1,
            max_depth=trial.suggest_int(
                name="max_depth",
                low=self.final_hyperparameter_search_ranges["max_depth"]["low"],
                high=self.final_hyperparameter_search_ranges["max_depth"]["high"],
            ),
            min_child_samples=trial.suggest_int(
                name="min_child_samples",
                low=self.final_hyperparameter_search_ranges["min_child_samples"]["low"],
                high=self.final_hyperparameter_search_ranges["min_child_samples"][
                    "high"
                ],
            ),
            learning_rate=trial.suggest_float(
                name="learning_rate",
                low=self.final_hyperparameter_search_ranges["learning_rate"]["low"],
                high=self.final_hyperparameter_search_ranges["learning_rate"]["high"],
                log=self.final_hyperparameter_search_ranges["learning_rate"]["log"],
            ),
            num_leaves=trial.suggest_int(
                name="num_leaves",
                low=self.final_hyperparameter_search_ranges["num_leaves"]["low"],
                high=self.final_hyperparameter_search_ranges["num_leaves"]["high"],
            ),
            n_estimators=trial.suggest_int(
                name="n_estimators",
                low=self.final_hyperparameter_search_ranges["n_estimators"]["low"],
                high=self.final_hyperparameter_search_ranges["n_estimators"]["high"],
                log=self.final_hyperparameter_search_ranges["n_estimators"]["log"],
            ),
            min_split_gain=trial.suggest_float(
                name="min_split_gain",
                low=self.final_hyperparameter_search_ranges["min_split_gain"]["low"],
                high=self.final_hyperparameter_search_ranges["min_split_gain"]["high"],
                log=self.final_hyperparameter_search_ranges["min_split_gain"]["log"],
            ),
            min_child_weight=trial.suggest_float(
                name="min_child_weight",
                low=self.final_hyperparameter_search_ranges["min_child_weight"]["low"],
                high=self.final_hyperparameter_search_ranges["min_child_weight"][
                    "high"
                ],
                log=self.final_hyperparameter_search_ranges["min_child_weight"]["log"],
            ),
            reg_alpha=trial.suggest_float(
                name="reg_alpha",
                low=self.final_hyperparameter_search_ranges["reg_alpha"]["low"],
                high=self.final_hyperparameter_search_ranges["reg_alpha"]["high"],
                log=self.final_hyperparameter_search_ranges["reg_alpha"]["log"],
            ),
            reg_lambda=trial.suggest_float(
                name="reg_lambda",
                low=self.final_hyperparameter_search_ranges["reg_lambda"]["low"],
                high=self.final_hyperparameter_search_ranges["reg_lambda"]["high"],
                log=self.final_hyperparameter_search_ranges["reg_lambda"]["log"],
            ),
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

        # DEFAULT RANGES FOR XBGPARAMS
        # "reg_lambda": {"low": 1e-4, "high": 100, "log": True},  # noqa: ERA001
        # "reg_alpha": {"low": 1e-4, "high": 100, "log": True},  # noqa: ERA001
        # "learning_rate": {"low": 1e-2, "high": 1, "log": True},  # noqa: ERA001
        # "max_depth": {"low": 1, "high": 5, "log": False},  # noqa: ERA001

        return XGBParams(
            verbosity=0,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=self.positive_class_weight,
            n_estimators=1000,
            colsample_bytree=1,
            n_jobs=1,
            reg_lambda=trial.suggest_float(
                name="reg_lambda",
                low=self.final_hyperparameter_search_ranges["reg_lambda"]["low"],
                high=self.final_hyperparameter_search_ranges["reg_lambda"]["high"],
                log=self.final_hyperparameter_search_ranges["reg_lambda"]["log"],
            ),
            reg_alpha=trial.suggest_float(
                name="reg_alpha",
                low=self.final_hyperparameter_search_ranges["reg_alpha"]["low"],
                high=self.final_hyperparameter_search_ranges["reg_alpha"]["high"],
                log=self.final_hyperparameter_search_ranges["reg_alpha"]["log"],
            ),
            learning_rate=trial.suggest_float(
                name="learning_rate",
                low=self.final_hyperparameter_search_ranges["learning_rate"]["low"],
                high=self.final_hyperparameter_search_ranges["learning_rate"]["high"],
                log=self.final_hyperparameter_search_ranges["learning_rate"]["log"],
            ),
            max_depth=trial.suggest_int(
                name="max_depth",
                low=self.final_hyperparameter_search_ranges["max_depth"]["low"],
                high=self.final_hyperparameter_search_ranges["max_depth"]["high"],
            ),
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
        # Default ranges for SVCParams
        #     "C": {"low": 1e-3, "high": 10000, "log": True},  # noqa: ERA001

        # TO DO: Sort these params out
        return SVCParams(
            class_weight={1: self.positive_class_weight, 0: 1},
            cache_size=1000,
            probability=False,
            kernel="rbf",
            shrinking=True,
            tol=1e-8,
            C=trial.suggest_float(
                name="C",
                low=self.final_hyperparameter_search_ranges["C"]["low"],
                high=self.final_hyperparameter_search_ranges["C"]["high"],
                log=self.final_hyperparameter_search_ranges["C"]["log"],
            ),
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
        # Random Forest default hyperparameter ranges
        #     "n_estimators": {"low": 100, "high": 1000, "log": False},  # noqa: ERA001

        return RandForestParams(
            verbose=0,
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
            n_estimators=trial.suggest_int(
                name="n_estimators",
                low=self.final_hyperparameter_search_ranges["n_estimators"]["low"],
                high=self.final_hyperparameter_search_ranges["n_estimators"]["high"],
            ),
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

    def optimisation_process_completed_callback(self, study, trial):
        """
        Stop all processes when one process stops searching.

        To run the optuna search, we spawn mulitple processes.
        In order to implement early stopping we use callbacks from optuna.
        These callbacks call study.stop(), which only stops trials spawned
        by the current process. This means that if one of the trials from the other
        processes breaks the early stopping condition, the search will not stop.
        This can result in a hyperparameter search with a single process running.
        To combat this, we use this callback to end all processes when one process calls
        study.stop().

        """
        if self.stopping_event.is_set():
            print("Ending process, stopping_event set.")
            study.stop()

    def create_search_callbacks(self) -> list:
        callbacks = []
        callbacks.append(self.optimisation_process_completed_callback)

        # Include the correct callbacks, based on user input, for stopping search
        if self.max_stagnation_iterations is not None:
            terminator = Terminator(
                improvement_evaluator=BestValueStagnationEvaluator(
                    max_stagnation_trials=self.max_stagnation_iterations
                ),
                error_evaluator=StaticErrorEvaluator(constant=0),
                min_n_trials=50,
            )
            stagnation_termination_callback = TerminatorCallback(terminator=terminator)
            callbacks.append(stagnation_termination_callback)

        if self.use_early_terminator:
            terminator = Terminator(
                improvement_evaluator=RegretBoundEvaluator(),
                error_evaluator=CrossValidationErrorEvaluator(),
                min_n_trials=50,
            )
            regret_terminator_callback = TerminatorCallback(terminator=terminator)
            callbacks.append(regret_terminator_callback)

        if self.max_n_search_iterations is not None:
            max_trials_callback = MaxTrialsCallback(
                self.max_n_search_iterations,
                states=(TrialState.COMPLETE, TrialState.PRUNED),
            )
            callbacks.append(max_trials_callback)

        return callbacks

    def create_best_cv_scores_shared_memory(self) -> None:
        array = np.zeros(self.num_cv_repeats * self.nfolds, dtype=np.float32)
        shm = multiprocessing.shared_memory.SharedMemory(create=True, size=array.nbytes)
        best_scores = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        best_scores[:] = array
        self.shm_name = shm.name
        self.shm_array_shape = array.shape
        shm.close()

    # Getter instead of a property to make it clear that this is going on under the hood
    def get_shared_memory_best_cv_scores(self):
        existing_shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
        current_best_scores = np.ndarray(
            self.shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
        )
        current_best_scores = np.array(current_best_scores)
        existing_shm.close()
        return current_best_scores

    def update_shared_memory_best_cv_scores(self, scores):
        existing_shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
        current_best_scores = np.ndarray(
            self.shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
        )
        if np.mean(current_best_scores) < np.mean(scores):
            current_best_scores[:] = scores
        existing_shm.close()

    def delete_best_cv_scores_shared_memory(self) -> None:
        existing_shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
        existing_shm.unlink()

    def wilcoxon_prune(
        self,
        trial,
        best_step_values,
        step_values,
    ) -> bool:
        if trial.number < 2:
            return False

        best_step_values = np.array(best_step_values)
        step_values = np.array(step_values)

        steps = np.array(range(step_values.shape[0]))

        diff_values = step_values[steps] - best_step_values[steps]

        # TO DO: GPT reccomended this, can we change?
        if len(diff_values) == 0:
            return False  # or handle this situation as needed

        alt = "less"
        average_is_best = sum(best_step_values) / len(best_step_values) <= sum(
            step_values
        ) / len(step_values)

        # We use zsplit to avoid the problem when all values are zero.
        p = ss.wilcoxon(diff_values, alternative=alt, zero_method="zsplit").pvalue

        if p < self.wilcoxon_trial_pruner_threshold and average_is_best:
            # ss.wilcoxon found the current trial is probably worse than the best trial,
            # but the value of the best trial was not better than
            # the average of the current trial's intermediate values.
            # For safety, WilcoxonPruner concludes not to prune it for now.
            return False
        return p < self.wilcoxon_trial_pruner_threshold

    def worse_than_first_two_prune(
        self,
        best_step_values,
        step_values,
    ) -> bool:
        # An easy way of preventing pruning of the first trial
        best_step_values = np.array(best_step_values)
        step_values = np.array(step_values)

        if len(step_values) > 1:
            return False

        return bool(
            step_values[0] < best_step_values[0]
            and step_values[0] < best_step_values[1]
        )

    def should_we_prune(self, trial, scores):
        current_best_scores = self.get_shared_memory_best_cv_scores()

        # Written this strange way because I think it improves readability
        if self.use_worse_than_first_two_pruner:
            should_prune = self.worse_than_first_two_prune(current_best_scores, scores)
            if should_prune:
                return True

        if self.wilcoxon_trial_pruner_threshold is not None:
            should_prune = self.wilcoxon_prune(trial, current_best_scores, scores)
            if should_prune:
                return True

        return False


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


def _predict_scores(
    model: LGBMClassifier | RandomForestClassifier | XGBClassifier | SVC,
    X: "csr_matrix",
) -> NDArray[np.float64] | NDArray[np.float32]:
    if isinstance(model, LGBMClassifier):
        return model.predict(X, raw_score=True)
    if isinstance(model, XGBClassifier):
        return model.predict(X, output_margin=True)
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
