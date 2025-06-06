"""Methods for optimising hyperparameters for models."""

import copy
from multiprocessing import cpu_count
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jsonpickle
import numpy as np
import optuna
from joblib import Parallel, delayed
from numpy.typing import NDArray

from . import validation
from .single_process_hparam_search import (
    SingleProcessHparamSearch,
)

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

# Considerations: Will the database work correctly in deployment?
# Considerations: Need a way to handle the interrupts
# Considerations: The cache size needs setting for the SVC
# Considerations: Should not use SVC for large datasets

# TO DO: Add a way for multiclass
# TO DO: Add a way to automatically fill the class weights for each objective function
# TO DO: Check the defaults are all good for the params
# TO DO: Fix all the params

# URGENT TO DO: MAKE SURE ALL THE MODELS A SINGLE CORE


# LIMIT DEFAULT MAX N_ESTIMATORS TO ABOUT 1000

# Verbosity, objective, n_jobs and scale_pos_weight cannot be changed

default_hyperparameter_ranges = {
    # TO DO: change default to linear
    "SVC": {
        # INTS
        "degree": {"low": 2, "high": 7, "log": False, "suggest_type": "int"},
        # FLOATS
        "C": {"low": 1e-3, "high": 100000, "log": True, "suggest_type": "float"},
        "gamma": {"low": 1e-7, "high": 1000, "log": True, "suggest_type": "float"},
        "coef0": {"low": 1e-7, "high": 1000, "log": True, "suggest_type": "float"},
        "tol": {"low": 1e-7, "high": 1e-3, "log": True, "suggest_type": "float"},
        # SINGULAR
        "cache_size": {"value": 1000, "suggest_type": "singular"},
        "shrinking": {"value": True, "suggest_type": "singular"},
        # CATEGORICAL
        "kernel": {
            "choices": [
                "linear",
                # "poly",
                "rbf",
                "sigmoid",
            ],
            "suggest_type": "categorical",
        },
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
        "max_depth": {"low": 1, "high": 60, "log": False, "suggest_type": "int"},
        "min_data_in_leaf": {
            "low": 1,
            "high": 30,
            "log": False,
            "suggest_type": "int",
        },
        "num_leaves": {"low": 2, "high": 70, "log": False, "suggest_type": "int"},
        "num_iterations": {
            "low": 100,
            "high": 2000,
            "log": False,
            "suggest_type": "int",
        },
        # FLOATS
        "learning_rate": {
            "low": 0.1,
            "high": 0.45,
            "log": False,
            "suggest_type": "float",
        },
        "min_gain_to_split": {
            "low": 1e-8,
            "high": 1,
            "log": True,
            "suggest_type": "float",
        },
        "min_sum_hessian_in_leaf": {
            "low": 1e-8,
            "high": 1e-2,
            "log": True,
            "suggest_type": "float",
        },
        "lambda_l1": {"low": 1e-9, "high": 1, "log": True, "suggest_type": "float"},
        "lambda_l2": {"low": 1e-9, "high": 1, "log": True, "suggest_type": "float"},
        # "path_smooth": {"low": 1e-6, "high": 0.5, "log": True, "suggest_type": "float"},
        # SINGULAR
        "data_sample_strategy": {"value": "bagging", "suggest_type": "singular"},
        "boosting": {"value": "gbdt", "suggest_type": "singular"},
        "tree_learner": {"value": "serial", "suggest_type": "singular"},
        "use_quantized_grad": {"value": False, "suggest_type": "singular"},
        "bagging_fraction": {"value": 1.0, "suggest_type": "singular"},
        "bin_construct_sample_cnt": {"value": 20000, "suggest_type": "singular"},
        "bagging_freq": {"value": 1, "suggest_type": "singular"},
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


class MultiProcessHparamSearch:
    """An engine for optimising hyperparameters for a using optuna."""

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
            Number of folds to use when performing cross-validation for evaluating
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

        if max_n_search_iterations <= 0:
            max_n_search_iterations = None
        if max_stagnation_iterations <= 0:
            max_stagnation_iterations = None
        if wilcoxon_trial_pruner_threshold <= 0:
            wilcoxon_trial_pruner_threshold = None

        # Bool to track if we need to use a pruner
        self.use_pruner = (
            self.wilcoxon_trial_pruner_threshold is not None
            or self.use_worse_than_first_two_pruner
        )
        print(f"self.use_pruner: {self.use_pruner}")

        if n_jobs == -1:
            self.n_jobs = cpu_count()
        else:
            self.n_jobs = n_jobs
        print(f"Number of processes: {self.n_jobs}")

        self.db_storage_url = self.select_database_url(db_url)

        self.final_hyperparameter_search_ranges = (
            self.define_hyperparameter_search_ranges(
                user_selected_hyperparameter_ranges, model_name
            )
        )

        self.positive_class_weight = np.count_nonzero(labels == 0) / np.count_nonzero(
            labels == 1
        )
        print(f"Positive class weight: {self.positive_class_weight}")

    def select_database_url(self, db_url: str | None) -> None:
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

        print(f"final ranges: {final_ranges}")
        return final_ranges

    def run_hparam_search_study(
        self,
        study_name: str,
    ) -> dict[str, Any]:
        """
        Initiate the hyperparameter search.

        Parameters
        ----------
        study_name : str
            A name to assign to a hyperparameter search. Allows for stopping
            and continuing the search at a later time.

        Returns
        -------
        dict
            Model hyperparameters that resulted in best cross-validation performance
            during the search. Key: hyperparameter name, value: hyperparameter value.

        """
        stopping_shm_name = create_stopping_event_shared_memory()

        cv_scores_shm_name = cv_scores_shm_shp = None
        if self.use_pruner:
            print(f"self.use_pruner for creating shared memory: {self.use_pruner}")
            print(
                f"type of self.use_pruner for creating shared_memory: {type(self.use_pruner)}"
            )
            print("we are creating the shared memory for pruning")
            # A pruner must be able to share the best scores between processes
            cv_scores_shm_name, cv_scores_shm_shp = create_best_cv_scores_shared_memory(
                num_cv_repeats=self.num_cv_repeats, nfolds=self.nfolds
            )

        storage_db = optuna.storages.RDBStorage(
            url=self.db_storage_url,
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_db,
            direction="maximize",
            load_if_exists=True,
        )
        try:
            # TO DO: try and remove the square brackets
            Parallel(n_jobs=self.n_jobs)(
                [
                    delayed(optimise_on_single)(
                        study_name,
                        tfidf_scores=self.tfidf_scores,
                        labels=self.labels,
                        nfolds=self.nfolds,
                        num_cv_repeats=self.num_cv_repeats,
                        timeout=self.timeout,
                        use_early_terminator=self.use_early_terminator,
                        max_stagnation_iterations=self.max_stagnation_iterations,
                        max_n_search_iterations=self.max_n_search_iterations,
                        wilcoxon_trial_pruner_threshold=self.wilcoxon_trial_pruner_threshold,
                        use_worse_than_first_two_pruner=self.use_worse_than_first_two_pruner,
                        model_name=self.model_name,
                        use_pruner=self.use_pruner,
                        db_storage_url=self.db_storage_url,
                        final_hyperparameter_search_ranges=self.final_hyperparameter_search_ranges,
                        positive_class_weight=self.positive_class_weight,
                        stopping_shm_name=stopping_shm_name,
                        cv_scores_shm_name=cv_scores_shm_name,
                        cv_scores_shm_shape=cv_scores_shm_shp,
                    )
                    for _ in range(self.n_jobs)
                ]
            )
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")

        if self.use_pruner:
            print(f"self.use_pruner for deleting shared memory: {self.use_pruner}")
            print("we are deleting the shared memory for pruning")
            # Once the search is complete, we must clean up the shared memory
            delete_shared_memory(cv_scores_shm_name)

        # We have shared memory to manage stopping needs to be removed
        delete_shared_memory(stopping_shm_name)

        return study

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


def create_stopping_event_shared_memory() -> str:
    shm = SharedMemory(create=True, size=np.bool_(False).nbytes)
    stopping_event = np.ndarray((1,), dtype=np.bool_, buffer=shm.buf)
    stopping_event[0] = False
    stopping_shm_name = shm.name
    shm.close()
    return stopping_shm_name


def create_best_cv_scores_shared_memory(num_cv_repeats: int, nfolds: int) -> None:
    array = np.zeros(num_cv_repeats * nfolds, dtype=np.float32)
    shm = SharedMemory(create=True, size=array.nbytes)
    best_scores = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    best_scores[:] = array
    shm_name = shm.name
    shm_array_shape = array.shape
    shm.close()
    return shm_name, shm_array_shape


def delete_shared_memory(shm_name: str) -> None:
    existing_shm = SharedMemory(name=shm_name)
    existing_shm.close()
    existing_shm.unlink()


def optimise_on_single(
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
) -> None:
    optimiser = SingleProcessHparamSearch(
        study_name=study_name,
        tfidf_scores=tfidf_scores,
        labels=labels,
        nfolds=nfolds,
        num_cv_repeats=num_cv_repeats,
        timeout=timeout,
        use_early_terminator=use_early_terminator,
        max_stagnation_iterations=max_stagnation_iterations,
        max_n_search_iterations=max_n_search_iterations,
        wilcoxon_trial_pruner_threshold=wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=use_worse_than_first_two_pruner,
        model_name=model_name,
        use_pruner=use_pruner,
        db_storage_url=db_storage_url,
        final_hyperparameter_search_ranges=final_hyperparameter_search_ranges,
        positive_class_weight=positive_class_weight,
        stopping_shm_name=stopping_shm_name,
        cv_scores_shm_name=cv_scores_shm_name,
        cv_scores_shm_shape=cv_scores_shm_shape,
    )
    optimiser.optimise()


def get_best_hparams_from_study(study: optuna.study.Study) -> dict[str, Any]:
    """
    Get the best hyperparameters from an optuna study.

    Parameters
    ----------
    study : optuna.study.Study
        An optuna study object.

    Returns
    -------
    dict
        Best hyperparameters found during the search. Key: hyperparameter name,
        value: hyperparameter value.

    """
    best_trial = study.best_trial
    best_params = best_trial.user_attrs["all_params"]
    return jsonpickle.decode(best_params, keys=True)
