import shutil
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
        "--resume_search_db",
        type=str,
        help="path to a database for which a hyperparameter search can be resumed",
        default=None,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="model name",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        help="number of cross-validation folds",
        default=3,
    )
    parser.add_argument(
        "--num_cv_repeats",
        type=int,
        help="number of times to repeat cross-validation for averaging scores",
        default=1,
    )
    parser.add_argument(
        "--user_selected_hyperparameter_ranges",
        type=str,
        help="dict of hyperparameter ranges and whether to use log search",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="time in seconds before cancelling search",
        default=None,
    )
    parser.add_argument(
        "--use_early_terminator",
        type=bool,
        help="whether to use the regret based early terminator",
        default=False,
    )
    parser.add_argument(
        "--max_stagnation_iterations",
        type=int,
        help="number of iterations of stagnation before cancelling search",
        default=None,
    )
    parser.add_argument(
        "--wilcoxon_trial_pruner_threshold",
        type=float,
        help="The p value for pruning search iterations based on wilcoxon signed rank test between current cross vals and best",
        default=None,
    )
    parser.add_argument(
        "--use_worse_than_first_two_pruner",
        type=bool,
        help="Bool wether to use custom pruner",
        default=False,
    )
    parser.add_argument(
        "--max_n_search_iterations",
        type=int,
        help="max number of iterations of hyperaparameter search to perform",
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

    if args.resume_search_db is None:
        optuna_db_path = os.path.join(args.search_db, "optuna.db")
        print(f"optuna_db_path: {optuna_db_path}")
    else:
        optuna_db_path = args.resume_search_db
    # with open("/mnt/optuna.db", 'w') as f:
    #     pass

    # We do this to turn the string into a dict
    user_selected_hyperparameter_ranges = args.user_selected_hyperparameter_ranges
    if user_selected_hyperparameter_ranges is not None:
        user_selected_hyperparameter_ranges = json.loads(
            user_selected_hyperparameter_ranges
        )

    print(f"model_name: {args.model_name}")
    print(f"n_folds: {args.n_folds}")
    print(f"num_cv_repeats: {args.num_cv_repeats}")
    print(f"max_n_search_iterations: {args.max_n_search_iterations}")
    print(f"user_selected_hyperparameter_ranges: {user_selected_hyperparameter_ranges}")
    print(f"timeout: {args.timeout}")
    print(f"use_early_terminator: {args.use_early_terminator}")
    print(f"max_stagnation_iterations: {args.max_stagnation_iterations}")
    print(f"wilcoxon_trial_pruner_threshold: {args.wilcoxon_trial_pruner_threshold}")
    print(f"use_worse_than_first_two_pruner: {args.use_worse_than_first_two_pruner}")

    # Perform the search
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores,
        labels,
        model_name=args.model_name,
        max_n_search_iterations=args.max_n_search_iterations,
        n_jobs=-1,
        nfolds=args.n_folds,
        num_cv_repeats=args.num_cv_repeats,
        # db_url=f"sqlite:////mnt/optuna.db", #Use this one on Azure
        # db_url=None,
        db_url=f"sqlite:///{optuna_db_path}",
        user_selected_hyperparameter_ranges=user_selected_hyperparameter_ranges,
        timeout=args.timeout,
        use_early_terminator=args.use_early_terminator,
        max_stagnation_iterations=args.max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=args.wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=args.use_worse_than_first_two_pruner,
    )

    start = time.time()
    best_params = optimiser.optimise_hyperparameters(study_name="hyperparam_search")
    print(f"Time taken: {time.time() - start}")

    # Save the best parameters
    best_params["model_name"] = args.model_name
    best_params = jsonpickle.encode(best_params, keys=True)
    best_param_path = os.path.join(args.best_params, "model_params.json")
    with open(best_param_path, "w") as f:
        json.dump(best_params, f)

    # If no database path was given, the search is automatically saved to the output db.
    # If a resume database was given, this does not happen, so we assgin it manually.
    if args.resume_search_db is not None:
        output_db_path = os.path.join(args.search_db, "optuna.db")
        shutil.copy(optuna_db_path, output_db_path)


if __name__ == "__main__":
    main()
