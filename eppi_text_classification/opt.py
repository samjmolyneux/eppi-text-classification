"""Methods for optimsing hyperparameters for models."""

import copy
import warnings
from dataclasses import asdict, dataclass, field
from multiprocessing import cpu_count
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonpickle
import lightgbm as lgb
import numpy as np
import optuna

# from optuna._imports import _LazyImport
import scipy.stats as ss
import xgboost as xgb
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
from .utils import SuppressStderr

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
    # TO DO: change default to linear
    "SVC": {
        # INTS
        # FLOATS
        "C": {"low": 1e-3, "high": 100000, "log": True},
        "gamma": {"low": 1e-7, "high": 1000, "log": True},
        # SINGULAR
        "cache_size": {"value": 1000, "suggest_type": "singular"},
        "kernel": {"value": "rbf", "suggest_type": "singular"},
        "shrinking": {"value": True, "suggest_type": "singular"},
        "tol": {"value": 1e-5, "suggest_type": "singular"},
        # CATEGORICAL
    },
    "xgboost": {
        # INTS
        "n_estimators": {"low": 100, "high": 1000, "log": False, "suggest_type": "int"},
        "max_depth": {"low": 1, "high": 5, "log": False, "suggest_type": "int"},
        # FLOATS
        "reg_lambda": {"low": 1e-4, "high": 100, "log": True, "suggest_type": "float"},
        "reg_alpha": {"low": 1e-4, "high": 100, "log": True, "suggest_type": "float"},
        "learning_rate": {"low": 1e-2, "high": 1, "log": True, "suggest_type": "float"},
        # SINGULAR
        "booster": {"value": "gbtree", "suggest_type": "singular"},
        "tree_method": {"value": "approx", "suggest_type": "singular"},
        "feature_selector": {"value": "cyclic", "suggest_type": "singular"},
        "verbosity": {"value": 2, "suggest_type": "singular"},
        # CATEGORICAL
    },
    "lightgbm": {
        # INTS
        "max_depth": {"low": 1, "high": 15, "log": False, "suggest_type": "int"},
        "min_child_samples": {
            "low": 1,
            "high": 30,
            "log": False,
            "suggest_type": "int",
        },
        "num_leaves": {"low": 2, "high": 50, "log": False, "suggest_type": "int"},
        "n_estimators": {"low": 100, "high": 1000, "log": False, "suggest_type": "int"},
        # FLOATS
        "learning_rate": {
            "low": 0.1,
            "high": 0.6,
            "log": False,
            "suggest_type": "float",
        },
        "min_split_gain": {
            "low": 1e-6,
            "high": 10,
            "log": True,
            "suggest_type": "float",
        },
        "min_child_weight": {
            "low": 1e-6,
            "high": 1e-1,
            "log": True,
            "suggest_type": "float",
        },
        "reg_alpha": {"low": 1e-5, "high": 10, "log": True, "suggest_type": "float"},
        "reg_lambda": {"low": 1e-5, "high": 10, "log": True, "suggest_type": "float"},
        # SINGULAR
        "data_sample_strategy": {"value": "bagging", "suggest_type": "singular"},
        "boosting_type": {"value": "gbdt", "suggest_type": "singular"},
        "tree_learner": {"value": "serial", "suggest_type": "singular"},
        "use_quantized_grad": {"value": False, "suggest_type": "singular"},
        "subsample": {"value": 1.0, "suggest_type": "singular"},
        "subsample_for_bin": {"value": 20000, "suggest_type": "singular"},
        # CATEGORICAL
    },
    "RandomForestClassifier": {
        # INTS
        "n_estimators": {
            "low": 100,
            "high": 1000,
            "log": False,
            "suggest_type": "int",
        },
        # SINGULAR
        "criterion": {"value": "gini", "suggest_type": "singular"},
        "max_depth": {"value": None, "suggest_type": "singular"},
        "max_features": {"value": "sqrt", "suggest_type": "singular"},
        "max_leaf_nodes": {"value": None, "suggest_type": "singular"},
        "bootstrap": {"value": True, "suggest_type": "singular"},
        "max_samples": {"value": None, "suggest_type": "singular"},
        "monotonic_cst": {"value": None, "suggest_type": "singular"},
        "min_samples_split": {"value": 2, "suggest_type": "singular"},
        "min_samples_leaf": {"value": 1, "suggest_type": "singular"},
        "min_weight_fraction_leaf": {"value": 0, "suggest_type": "singular"},
        "min_impurity_decrease": {"value": 0, "suggest_type": "singular"},
        "ccp_alpha": {"value": 0, "suggest_type": "singular"},
        # CATEGORICAL
        # "max_depth": {"choices": [None, 5, 10, 15, 20], "suggest_type": "categorical"},
    },
}

# STILL NEED TO DOUBLE CHECK HYPERPARAMETER RANGES

model_hyperparameter_dependencies = {
    "SVC": {
        "degree": {"kernel": ["poly"]},
        "coef0": {"kernel": ["poly", "sigmoid"]},
        "gamma": {"kernel": ["rbf", "poly", "sigmoid"]},
    },
    "xgboost": {
        "learning_rate": {"booster": ["gbtree", "dart"]},
        "gamma": {"booster": ["gbtree", "dart"]},
        "max_depth": {"booster": ["gbtree", "dart"]},
        "min_child_weight": {"booster": ["gbtree", "dart"]},
        "subsample": {"booster": ["gbtree", "dart"]},
        "max_delta_step": {"booster": ["gbtree", "dart"]},
        "tree_method": {"booster": ["gbtree", "dart"]},
        "grow_plolicy": {
            "booster": ["gbtree", "dart"],
            "tree_method": ["hist", "approx", "auto"],
        },
        "max_leaves": {
            "booster": ["gbtree", "dart"],
            "tree_method": ["hist", "approx", "auto"],
        },
        # ?
        "max_bin": {
            "booster": ["gbtree", "dart"],
            "tree_method": ["hist", "approx", "auto"],
        },
        "num_parallel_tree": {"booster": ["gbtree", "dart"]},
        "sample_type": {"booster": ["dart"]},
        "normalize_type": {"booster": ["dart"]},
        "rate_drop": {"booster": ["dart"]},
        "one_drop": {"booster": ["dart"]},
        "skip_drop": {"booster": ["dart"]},
        "updater": {"booster": ["gblinear"]},
        "feature_selector": {"booster": ["gblinear"]},
        "top_k": {
            "booster": ["gblinear"],
            "feature_selector": ["greedy", "thrifty"],
        },
    },
    "lightgbm": {
        "bagging_freq": {"data_sample_strategy": ["bagging"]},
        "bagging_fraction": {"data_sample_strategy": ["bagging"]},
        "drop_rate": {"boosting_type": ["dart"]},
        "max_drop": {"boosting_type": ["dart"]},
        "skip_drop": {"boosting_type": ["dart"]},
        "xgboost_dart_mode": {"boosting_type": ["dart"]},
        "uniform_drop": {"boosting_type": ["dart"]},
        "top_rate": {"data_sample_strategy": ["goss"]},
        "other_rate": {"data_sample_strategy": ["goss"]},
        "top_k": {"tree_learner": ["voting"]},
        "num_grad_quant_bins": {"use_quantized_grad": [True]},
        "quant_train_renew_leaf": {"use_quantized_grad": [True]},
        "stochastic_rounding": {"use_quantized_grad": [True]},
    },
    "RandomForestClassifier": {
        "oob_score": {"bootstrap": [True]},
        "max_samples": {"bootstrap": [True]},
    },
}

model_name_to_selector = {
    "SVC": "select_svc_hyperparameters",
    "lightgbm": "select_lgbm_hyperparameters",
    "RandomForestClassifier": "select_rand_forest_hyperparameters",
    "xgboost": "select_xgb_hyperparameters",
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
        self.model_name = model_name

        # Bool to track if we need to use a pruner
        self.use_pruner = (
            self.wilcoxon_trial_pruner_threshold is not None
            or self.use_worse_than_first_two_pruner
        )

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        print(f"Number of processes: {self.n_jobs}")

        self.select_hyperparameters = getattr(self, model_name_to_selector[model_name])

        self.db_storage_url = self.set_database_url(db_url)

        self.final_hyperparameter_search_ranges = (
            self.define_hyperparameter_search_ranges(
                user_selected_hyperparameter_ranges, model_name
            )
        )

        self.positive_class_weight = np.count_nonzero(labels == 0) / np.count_nonzero(
            labels == 1
        )
        print(f"Positive class weight: {self.positive_class_weight}")

    def set_database_url(self, db_url: str | None) -> None:
        """
        Set up the database for the hyperparameter search.

        Parameters
        ----------
        db_url :
            URL to the database to store the hyperparameter search history.

        """
        if db_url is None:
            root_path = Path(Path(__file__).resolve()).parent.parent
            db_storage_url = f"sqlite:///{root_path}/optuna.db"
        else:
            db_storage_url = db_url

        validation.check_valid_database_url(db_storage_url)

        # If a database does not exist, it will be created by optuna.
        print(db_storage_url)

        return db_storage_url

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
        default_ranges = default_hyperparameter_ranges[model_name]

        if user_selected_ranges is None:
            return default_ranges

        final_ranges = copy.deepcopy(default_ranges)
        final_ranges.update(user_selected_ranges)

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
        self.create_stopping_event_shared_memory()

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

                clf = _train_model(self.model_name, params, X_train, y_train)

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
        # We must first select values for parameters that have other parameters dependent on them.
        params = self.suggest_hyperparams_from_ranges(
            trial, self.final_hyperparameter_search_ranges
        )

        # Do not use class_weight, force_col_wise, subsample, n_jobs, device,
        # scale_pos_weight, is_unbalance, early_stopping_rounds, eval_metric,
        # early_stopping_min_delta, xgboost_dart_mode, min_data_per_group,
        # max_cat_threshold, cat_l2, cat_smooth, max_cat_to_onehot,
        # monotone_constraints, monotone_constraints_method, monotone_penalty,
        # feature_contri, forced_splits_filename, refit_decay_rate,
        # cegb_penalty_feature_lazy, cegb_penalty_feature_coupled, path_smooth,
        # interaction_constraints, verbosity, snapshot_freq,
        # saved_feature_importance_type, use_quantized_grad,
        # num_grad_quant_bins, quant_train_renwew_leaf, stochastic_rounding,
        # max_bin_by_feature, is_enable_sparse, enable_bundle, use_missing,
        # zero_as_missing, pre_partition, use_two_round_loading, header,
        # label_column, weight_column, group_column, ignore_column,
        # categegorical_feature, forcedbins_filename, save_binary,
        # precise_float_parser, parser_config_file
        return {
            "verbosity": 0,
            "objective": "binary",
            "scale_pos_weight": self.positive_class_weight,
            "n_jobs": 1,
            "device": "cpu",
            "force_col_wise": False,
            "force_row_wise": False,
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

        # Also, dont use max_cat_to_onehot, max_cat_threshold, multi_strategy,
        # early_stopping_rounds, eval_metric, callbacks, process_type,
        # colsample_by*, refresh_leaf, max_cached_hist_node,
        return {
            "verbosity": 2,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "scale_pos_weight": self.positive_class_weight,
            "colsample_bytree": 1,
            "n_jobs": 1,
            "device": "cpu",
            "monotone_constraints": None,
            "interaction_constraints": None,
            "enable_categorical": False,
            "feature_types": None,
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
            "verbose": False,
            "decision_function_shape": "ovr",
            "break_ties": False,
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

        # Dont use monotonic_cst
        return {
            "verbose": 0,
            "n_jobs": 1,
            "class_weight": {1: self.positive_class_weight, 0: 1},
            "warm_start": False,
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
        trial: optuna.trial.Trial,
        hyperparameter_search_dict: dict[str, dict],
    ) -> dict[str, Any]:
        final_hyperparameter_values = {}
        unassigned_hyperparams = set(hyperparameter_search_dict.keys())

        while len(unassigned_hyperparams) > 0:
            progress_flag = False

            for hyperparameter in list(unassigned_hyperparams):
                param_dependencies_dict = model_hyperparameter_dependencies[
                    self.model_name
                ].get(hyperparameter, {})

                # We can set the param if all parent keys satisfy dependent condition
                if all(
                    dependency in final_hyperparameter_values
                    for dependency in param_dependencies_dict
                ):
                    if all(
                        final_hyperparameter_values[dependency] in valid_values
                        for dependency, valid_values in param_dependencies_dict.items()
                    ):
                        if hyperparameter == "oob_score":
                            print("inside")
                        param_range_dict = hyperparameter_search_dict[hyperparameter]

                        # Otherwise, this param should be set, because it is either not dependent on
                        # other params or the parent params have been assigned values that mean this
                        # param should be set
                        if param_range_dict["suggest_type"] == "singular":
                            final_hyperparameter_values[hyperparameter] = (
                                param_range_dict["value"]
                            )

                        elif param_range_dict["suggest_type"] == "int":
                            final_hyperparameter_values[hyperparameter] = (
                                trial.suggest_int(
                                    name=hyperparameter,
                                    low=param_range_dict["low"],
                                    high=param_range_dict["high"],
                                    log=param_range_dict["log"],
                                )
                            )

                        elif param_range_dict["suggest_type"] == "float":
                            final_hyperparameter_values[hyperparameter] = (
                                trial.suggest_float(
                                    name=hyperparameter,
                                    low=param_range_dict["low"],
                                    high=param_range_dict["high"],
                                    log=param_range_dict["log"],
                                )
                            )
                        # TO DO: Test this
                        elif param_range_dict["suggest_type"] == "categorical":
                            final_hyperparameter_values[hyperparameter] = (
                                trial.suggest_categorical(
                                    name=hyperparameter,
                                    choices=param_range_dict["choices"],
                                )
                            )

                        unassigned_hyperparams.remove(hyperparameter)
                        progress_flag = True

                    else:
                        # If we have all the dependencies, but the values are not valid, we must
                        # remove the hyperparameter from the list of unassigned hyperparameters
                        unassigned_hyperparams.remove(hyperparameter)
                        progress_flag = True

                elif any(
                    (
                        (dependency not in final_hyperparameter_values)
                        and (dependency not in unassigned_hyperparams)
                    )
                    for dependency in param_dependencies_dict
                ):
                    # If we are missing a dependency, we must remove the hyperparameter from the
                    # list of unassigned hyperparameters
                    unassigned_hyperparams.remove(hyperparameter)
                    progress_flag = True

            if progress_flag is False:
                msg = (
                    "Hyperaparameter suggestion failed a pass. This means "
                    "the dependencies have been set incorrecty "
                    "or the hyperparameter ranges are incorrect."
                )
                raise ValueError(msg)

        return final_hyperparameter_values


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
    if isinstance(model, lgb.basic.Booster):
        return model.predict(X, raw_score=True)
    if isinstance(model, xgb.core.Booster):
        dtrain = xgb.DMatrix(X)
        return model.predict(dtrain, output_margin=True)
    if isinstance(model, RandomForestClassifier):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _train_model(
    model_name: str,
    params: dict[str, Any],
    X: "csr_matrix",
    y: NDArray[np.int64],
) -> NDArray[np.float64] | NDArray[np.float32]:
    if model_name == "lightgbm":
        train_params = copy.deepcopy(params)
        with SuppressStderr(
            [
                "No further splits with positive gain, best gain: -inf",
                "Stopped training because there are no more leaves that meet the split requirements",
            ]
        ):
            num_boost_round = train_params.pop("n_estimators")
            dtrain = lgb.Dataset(X, label=y, free_raw_data=True)
            model = lgb.train(train_params, dtrain, num_boost_round=num_boost_round)
        return model

    if model_name == "xgboost":
        train_params = copy.deepcopy(params)
        num_boost_round = train_params.pop("n_estimators")
        # enable categorical is always false
        enable_categorical = train_params.pop("enable_categorical")
        dtrain = xgb.DMatrix(X, label=y, enable_categorical=enable_categorical)
        # xgboost.train ignores params it cannot use
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=num_boost_round,
        )
        # gblinear uses a different shap explainer, so we save booster info
        model.set_attr(boosting_type=params["booster"])
        return model

    if model_name == "RandomForestClassifier":
        model = RandomForestClassifier(**params)
        model.fit(X, y)
        return model

    if model_name == "SVC":
        model = SVC(**params)
        model.fit(X, y)
        return model

    print(f"model is {model}")
    msg = "Model not recognised."
    raise ValueError(msg)
