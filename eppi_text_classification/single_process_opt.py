import copy
import time

from multiprocessing.shared_memory import SharedMemory
from typing import TYPE_CHECKING, Any

import jsonpickle
import lightgbm as lgb
import numpy as np
import optuna
import scipy.stats as ss
import xgboost as xgb
from numpy.typing import NDArray
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
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from .utils import SuppressStderr

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

model_name_to_selector = {
    "SVC": "select_svc_hyperparameters",
    "lightgbm": "select_lgbm_hyperparameters",
    "RandomForestClassifier": "select_rand_forest_hyperparameters",
    "xgboost": "select_xgb_hyperparameters",
}

model_hyperparameter_dependencies = {
    "SVC": {
        "degree": {"kernel": ["poly"]},
        "coef0": {"kernel": ["poly", "sigmoid"]},
        "gamma": {"kernel": ["rbf", "poly", "sigmoid"]},
    },
    "xgboost": {
        # NEED TO REMOVE GBLINEAR FROM LEARNING RATE
        "learning_rate": {"booster": ["gbtree", "dart", "gblinear"]},
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


class SingleProcessHyperparameterOptimiser:
    def __init__(
        self,
        study_name,
        tfidf_scores,
        labels,
        nfolds,
        num_cv_repeats,
        timeout,
        use_early_terminator,
        max_stagnation_iterations,
        max_n_search_iterations,
        wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner,
        model_name,
        use_pruner,
        db_storage_url,
        final_hyperparameter_search_ranges,
        positive_class_weight,
        stopping_shm_name,
        cv_scores_shm_name,
        cv_scores_shm_shape,
    ):
        self.study_name = study_name
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
        self.use_pruner = use_pruner
        self.db_storage_url = db_storage_url
        self.final_hyperparameter_search_ranges = final_hyperparameter_search_ranges
        self.positive_class_weight = positive_class_weight
        self.stopping_shm_name = stopping_shm_name
        self.cv_scores_shm_name = cv_scores_shm_name
        self.cv_scores_shm_shape = cv_scores_shm_shape

        self.select_hyperparameters = getattr(self, model_name_to_selector[model_name])

    def optimise(self):
        study = optuna.load_study(
            study_name=self.study_name, storage=self.db_storage_url
        )

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
        set_stopping_event(self.stopping_shm_name)

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
            update_shared_memory_best_cv_scores(scores)

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
            # "colsample_bytree": 1,
            "n_jobs": 1,
            "device": "cpu",
            "monotone_constraints": None,
            "interaction_constraints": None,
            "enable_categorical": False,
            "feature_types": None,
            "validate_parameters": True,
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
        if is_stopping_event_set(self.stopping_shm_name):
            print("Ending process, stopping_event set.")
            study.stop()

    def create_search_callbacks(self) -> list:
        callbacks = []

        opt_finished_callback = self.optimisation_process_completed_callback
        callbacks.append(opt_finished_callback)

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
        current_best_scores = get_shared_memory_best_cv_scores()

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

                        # print(f"removing first else: {hyperparameter}")
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
                    # print(f"removing: {hyperparameter}")
                    unassigned_hyperparams.remove(hyperparameter)
                    progress_flag = True

            if progress_flag is False:
                msg = (
                    "Hyperaparameter suggestion failed a pass. This means "
                    "the dependencies have been set incorrecty "
                    "or the hyperparameter ranges are incorrect."
                )
                raise ValueError(msg)

        # print(f"time taken to suggest hyperparameters: {time.time() - start}")
        return final_hyperparameter_values


def set_stopping_event(stopping_shm_name: str) -> None:
    existing_shm = SharedMemory(name=stopping_shm_name)
    stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=existing_shm.buf)
    stopping_event[0] = True
    existing_shm.close()


def is_stopping_event_set(stopping_shm_name: str) -> bool:
    existing_shm = SharedMemory(name=stopping_shm_name)
    stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=existing_shm.buf)
    stopping_event_set_bool = stopping_event[0]
    existing_shm.close()

    return stopping_event_set_bool


# Getter instead of a property to make it clear that this is going on under the hood
def get_shared_memory_best_cv_scores(shm_name, shm_array_shape) -> NDArray[np.float32]:
    existing_shm = SharedMemory(name=shm_name)
    current_best_scores = np.ndarray(
        shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
    )
    current_best_scores = np.array(current_best_scores)
    existing_shm.close()
    return current_best_scores


def update_shared_memory_best_cv_scores(shm_name, shm_array_shape, scores):
    existing_shm = SharedMemory(name=shm_name)
    current_best_scores = np.ndarray(
        shm_array_shape, dtype=np.float32, buffer=existing_shm.buf
    )
    if np.mean(current_best_scores) < np.mean(scores):
        current_best_scores[:] = scores
    existing_shm.close()


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


def _predict_scores(
    model,
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
