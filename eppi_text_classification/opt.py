"""Methods for optimsing hyperparameters for models."""

from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from multiprocessing.shared_memory import SharedMemory
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


# LLIMIT DEFAULT MAX N_ESTIMSTORS TO ABOIUT 1000

# Verbosity, objective, n_jobs and scale_pos_weight cannot be changed

default_hyperparameter_ranges = {
    "SVC": {
        "suggest_float": {
            "C": {"low": 1e-3, "high": 100000, "log": True},
            "gamma": {"low": 1e-7, "high": 1000, "log": True},
        },
        "singular": {
            "cache_size": 1000,
            "kernel": "rbf",
            "shrinking": True,
            "tol": 1e-5,
        },
    },
    "XGBClassifier": {
        "suggest_int": {
            "n_estimators": {"low": 100, "high": 1000, "log": False},
            "max_depth": {"low": 1, "high": 5, "log": False},
        },
        "suggest_float": {
            "reg_lambda": {"low": 1e-4, "high": 100, "log": True},
            "reg_alpha": {"low": 1e-4, "high": 100, "log": True},
            "learning_rate": {"low": 1e-2, "high": 1, "log": True},
        },
        "singular": {
            "colsample_bytree": 1.0,
        },
    },
    "LGBMClassifier": {
        "suggest_int": {
            "max_depth": {"low": 1, "high": 15, "log": False},
            "min_child_samples": {"low": 1, "high": 30, "log": False},
            "num_leaves": {"low": 2, "high": 50, "log": False},
            "n_estimators": {"low": 100, "high": 1000, "log": False},
            "subsample_for_bin": {"low": 20000, "high": 20000},
        },
        "suggest_float": {
            "learning_rate": {"low": 0.1, "high": 0.6, "log": False},
            "min_split_gain": {"low": 1e-6, "high": 10, "log": True},
            "min_child_weight": {
                "low": 1e-6,
                "high": 1e-1,
                "log": True,
            },
            "reg_alpha": {"low": 1e-5, "high": 10, "log": True},
            "reg_lambda": {"low": 1e-5, "high": 10, "log": True},
        },
        "singular": {
            "subsample": 1.0,
            "boosting_type": "gbdt",
        },
    },
    "RandomForestClassifier": {
        "suggest_int": {
            "n_estimators": {
                "low": 100,
                "high": 1000,
                "log": False,
                "data_type": "float",
            },
        },
        "singular": {
            "criterion": "gini",
            "max_depth": None,
            "max_features": "sqrt",
            "max_leaf_nodes": None,
            "bootstrap": True,
            "max_samples": None,
            "monotonic_cst": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0,
            "min_impurity_decrease": 0,
            "ccp_alpha": 0,
        },
        # "Categorical" : { {param: [categories...]}
    },
}

# Verbosity, objective, n_jobs, probability and scale_pos_weight cannot be changed
# Verbosity, objective, n_jobs and class_weight cannot be changed
# Verbosity, objective, n_jobs and class_weight cannot be changed


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
        self.create_stopping_event_shared_memory()

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
        # NEED TO MAKE IT WORK WITH NEW SCHEMA
        default_ranges = default_hyperparameter_ranges[model_name]

        if user_selected_ranges is None:
            return default_ranges

        final_ranges = {}
        for suggest_type, hyperparameter_ranges in default_ranges.items():
            for param, param_default_range in hyperparameter_ranges.items():
                final_ranges[suggest_type][param] = user_selected_ranges.get(
                    param,
                    param_default_range,
                )

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
        self.set_stopping_event()

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

        # We have shared memory to manage stopping needs to be removed
        self.delete_stopping_event_shared_memory()

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
        serialized_params = jsonpickle.encode(params, keys=True)
        trial.set_user_attr("all_params", serialized_params)
        clf = self.model_class(**params)

        # Calculate the cross validation score
        scores = []
        for i in range(self.num_cv_repeats):
            kf = StratifiedKFold(n_splits=self.nfolds, shuffle=True, random_state=i)

            for _, (train_idx, val_idx) in enumerate(
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

    def select_lgbm_hyperparameters(self, trial: optuna.trial.Trial) -> dict:
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
        params = self.suggest_hyperparams_from_ranges(
            trial, self.final_hyperparameter_search_ranges
        )

        return {
            "verbosity": -1,
            "subsample": 1.0,
            "objective": "binary",
            "scale_pos_weight": self.positive_class_weight,
            "n_jobs": 1,
            **params,
        }

    def select_xgb_hyperparameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
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

        params = self.suggest_hyperparams_from_ranges(
            trial, self.final_hyperparameter_search_ranges
        )

        return {
            "verbosity": 0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": self.positive_class_weight,
            "colsample_bytree": 1,
            "n_jobs": 1,
            **params,
        }

    def select_svc_hyperparameters(self, trial: optuna.trial.Trial) -> dict[str, Any]:
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
        params = self.suggest_hyperparams_from_ranges(
            trial, self.final_hyperparameter_search_ranges
        )

        # TO DO: Sort these params out
        return {
            "class_weight": {1: self.positive_class_weight, 0: 1},
            "probability": False,
            **params,
        }

    def select_rand_forest_hyperparameters(
        self, trial: optuna.trial.Trial
    ) -> dict[str, Any]:
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
        params = self.suggest_hyperparams_from_ranges(
            trial, self.final_hyperparameter_search_ranges
        )

        return {
            "verbose": 0,
            "n_jobs": 1,
            "class_weight": {1: self.positive_class_weight, 0: 1},
            **params,
        }

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
        if self.is_stopping_event_set():
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

    def create_stopping_event_shared_memory(self) -> None:
        shm = SharedMemory(create=True, size=np.bool_(False).nbytes)
        stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=shm.buf)
        stopping_event[0] = False
        self.stopping_shm_name = shm.name
        shm.close()

    def set_stopping_event(self):
        existing_shm = SharedMemory(name=self.stopping_shm_name)
        stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=existing_shm.buf)
        stopping_event[0] = True
        existing_shm.close()

    def delete_stopping_event_shared_memory(self) -> None:
        existing_shm = SharedMemory(name=self.stopping_shm_name)
        existing_shm.close()
        existing_shm.unlink()

    def is_stopping_event_set(self):
        existing_shm = SharedMemory(name=self.stopping_shm_name)
        stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=existing_shm.buf)
        stopping_event_set_bool = stopping_event[0]
        existing_shm.close()

        return stopping_event_set_bool

    def create_best_cv_scores_shared_memory(self) -> None:
        array = np.zeros(self.num_cv_repeats * self.nfolds, dtype=np.float32)
        shm = SharedMemory(create=True, size=array.nbytes)
        best_scores = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        best_scores[:] = array
        self.shm_name = shm.name
        self.shm_array_shape = array.shape
        shm.close()

    # Getter instead of a property to make it clear that this is going on under the hood
    def get_shared_memory_best_cv_scores(self):
        existing_shm = SharedMemory(name=self.shm_name)
        current_best_scores = np.ndarray(
            self.shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
        )
        current_best_scores = np.array(current_best_scores)
        existing_shm.close()
        return current_best_scores

    def update_shared_memory_best_cv_scores(self, scores):
        existing_shm = SharedMemory(name=self.shm_name)
        current_best_scores = np.ndarray(
            self.shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
        )
        if np.mean(current_best_scores) < np.mean(scores):
            current_best_scores[:] = scores
        existing_shm.close()

    def delete_best_cv_scores_shared_memory(self) -> None:
        existing_shm = SharedMemory(name=self.shm_name)
        existing_shm.close()
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

    def suggest_hyperparams_from_ranges(
        self,
        trial,
        hyperparameter_search_dict: dict[str, dict],
    ) -> dict[str, Any]:
        hyperparameter_values = {}
        for suggest_type, hyperparameter_ranges in hyperparameter_search_dict.items():
            if suggest_type == "singular":
                hyperparameter_values.update(hyperparameter_ranges)

            elif suggest_type == "suggest_int":
                for param, params_range in hyperparameter_ranges.items():
                    hyperparameter_values[param] = trial.suggest_int(
                        name=param,
                        low=params_range["low"],
                        high=params_range["high"],
                    )

            elif suggest_type == "suggest_float":
                for param, params_range in hyperparameter_ranges.items():
                    hyperparameter_values[param] = trial.suggest_float(
                        name=param,
                        low=params_range["low"],
                        high=params_range["high"],
                        log=params_range["log"],
                    )
            # TO DO: Test this
            if suggest_type == "categorical":
                hyperparameter_values[param] = trial.suggest_categorical(
                    name=param,
                    choices=params_range["categories"],
                )

        return hyperparameter_values


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
