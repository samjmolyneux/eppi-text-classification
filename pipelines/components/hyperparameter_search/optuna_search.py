import argparse
import json
import os
import time

import jsonpickle

from eppi_text_classification import OptunaHyperparameterOptimisation
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_np_array_at_directory,
)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        help="path to ordered list of labels",
    )
    parser.add_argument(
        "--tfidf_scores",
        type=str,
        help="path to tfidf scores for data",
    )
    parser.add_argument(
        "--search_parameters",
        type=str,
        help="path to search parameters for the optuna search",
    )
    parser.add_argument(
        "--best_params",
        type=str,
        help="path to best hypereparameters found by the search",
    )
    parser.add_argument(
        "--search_db",
        type=str,
        help="path to optuna search database",
    )
    args = parser.parse_args()

    tfidf_scores = load_csr_at_directory(args.tfidf_scores)
    labels = load_np_array_at_directory(args.labels)
    with open(args.search_parameters, "r") as file:
        json_search_parameters = file.read()
    kwargs = jsonpickle.decode(json_search_parameters)

    optuna_db_path = os.path.join(args.search_db, "optuna.db")
    print(f"optuna_db_path: {optuna_db_path}")

    # with open("/mnt/optuna.db", 'w') as f:
    #     pass

    model_name = kwargs["model_name"]
    num_trials_per_job = kwargs["num_trials_per_job"]
    n_folds = 3 if "n_folds" not in kwargs else kwargs["n_folds"]
    num_cv_repeats = 1 if "num_cv_repeats" not in kwargs else kwargs["num_cv_repeats"]
    print(f"model_name: {model_name}")
    print(f"num_trials_per_job: {num_trials_per_job}")
    print(f"n_folds: {n_folds}")
    print(f"num_cv_repeats: {num_cv_repeats}")

    # Perform the search
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        model_name,
        n_trials_per_job=num_trials_per_job,
        n_jobs=-1,
        nfolds=n_folds,
        num_cv_repeats=num_cv_repeats,
        # db_url=f"sqlite:////mnt/optuna.db", #Use this one on Azure
        # db_url=None,
        db_url=f"sqlite:///{optuna_db_path}",
    )

    start = time.time()
    best_params = optimiser.optimise_hyperparameters(study_name="hyperparam_search")
    print(f"Time taken: {time.time() - start}")

    # Save the best parameters
    best_params["model_name"] = model_name
    best_params = jsonpickle.encode(best_params, keys=True)
    best_param_path = os.path.join(args.best_params, "model_params.json")
    with open(best_param_path, "w") as f:
        json.dump(best_params, f)


if __name__ == "__main__":
    main()
