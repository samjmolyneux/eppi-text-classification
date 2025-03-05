import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from eppi_text_classification import get_features, get_labels, get_tfidf_and_names
from eppi_text_classification.model_stability import (
    predict_cv_metrics,
    predict_cv_scores,
)
from eppi_text_classification.opt import (
    OptunaHyperparameterOptimisation,
    get_best_hyperparams_from_study,
)
from eppi_text_classification.plots import (
    box_plot,
    create_all_optuna_plots,
    histogram_plot,
    learning_curve,
    positive_negative_scores_histogram_plot,
    roc_plot,
    select_threshold_plot,
)
from eppi_text_classification.shap_plotter import ShapPlotter
from eppi_text_classification.train import train


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input tsv")
    parser.add_argument(
        "--title_header",
        type=str,
        help="Name of title column",
        default="PaperTitle",
    )
    parser.add_argument(
        "--abstract_header",
        type=str,
        help="Name of abstract column",
        default="Abstract",
    )
    parser.add_argument(
        "--label_header",
        type=str,
        help="Name of label column",
        default="Label",
    )
    parser.add_argument(
        "--positive_class_value",
        type=str,
        help="What value is given to the positive class in the dataset",
        default="1",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model used in the search",
    )
    parser.add_argument(
        "--hyperparameter_search_ranges",
        type=str,
        help="Path to directory containing hyperparameter search ranges json",
        default=None,
    )
    parser.add_argument(
        "--max_n_search_iterations",
        type=int,
        help="Maximum number of search iterations",
        # default=None,
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        help="Number of folds for cross-validation in the hyperparameter search",
        default=3,
    )
    parser.add_argument(
        "--num_cv_repeats",
        type=int,
        help="Number of times to average the cross-validation to stabilize the search",
        default=1,
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Time in seconds before cancelling search",
        default=None,
    )
    parser.add_argument(
        "--use_early_terminator",
        type=str,
        help="Whether to use the regret based early terminator",
        default=False,
    )
    parser.add_argument(
        "--max_stagnation_iterations",
        type=int,
        help="Number of iterations of stagnation before cancelling search",
        default=None,
    )
    parser.add_argument(
        "--wilcoxon_trial_pruner_threshold",
        type=float,
        help="P Value below which wilcoxon trial pruner will prune",
        default=None,
    )
    parser.add_argument(
        "--use_worse_than_first_two_pruner",
        type=str,
        help="Whether to use the worse than first two pruner",
        default=False,
    )
    parser.add_argument(
        "--shap_num_display",
        type=int,
        help="Number of features to display in shap plots",
        default=10,
    )
    parser.add_argument(
        "--search_db",
        type=str,
        help="Path to directory containing optuna search database",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="Path to feature names",
    )
    parser.add_argument(
        "--tfidf_scores",
        type=str,
        help="Path to tfidf scores",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to labels",
    )
    parser.add_argument(
        "--plots",
        type=str,
        help="Path to directory to plots",
    )
    parser.add_argument(
        "--best_hparams",
        type=str,
        help="Path to best hyperparameters",
    )
    args = parser.parse_args()

    data_path = args.data
    title_header = args.title_header
    abstract_header = args.abstract_header
    label_header = args.label_header
    positive_class_value = args.positive_class_value
    model_name = args.model_name
    max_n_search_iterations = args.max_n_search_iterations
    nfolds = args.nfolds
    num_cv_repeats = args.num_cv_repeats
    timeout = args.timeout
    use_early_terminator = args.use_early_terminator
    max_stagnation_iterations = args.max_stagnation_iterations
    wilcoxon_trial_pruner_threshold = args.wilcoxon_trial_pruner_threshold
    use_worse_than_first_two_pruner = args.use_worse_than_first_two_pruner
    shap_num_display = args.shap_num_display

    with open(args.hyperparameter_search_ranges) as f:
        hyperparameter_search_ranges = json.load(f)

    search_db_url = f"sqlite:///{args.search_db}/optuna.db"

    plots_directory = args.plots
    os.mkdir(f"{plots_directory}/optuna_plots")

    tprint(f"data_path: {data_path}")
    tprint(f"title_header: {title_header}")
    tprint(f"abstract_header: {abstract_header}")
    tprint(f"label_header: {label_header}")
    tprint(f"positive_class_value: {positive_class_value}")
    tprint(f"model_name: {model_name}")
    tprint(f"max_n_search_iterations: {max_n_search_iterations}")
    tprint(f"nfolds: {nfolds}")
    tprint(f"num_cv_repeats: {num_cv_repeats}")
    tprint(f"timeout: {timeout}")
    tprint(f"use_early_terminator: {use_early_terminator}")
    tprint(f"max_stagnation_iterations: {max_stagnation_iterations}")
    tprint(f"wilcoxon_trial_pruner_threshold: {wilcoxon_trial_pruner_threshold}")
    tprint(f"use_worse_than_first_two_pruner: {use_worse_than_first_two_pruner}")
    tprint(f"shap_num_display: {shap_num_display}")
    tprint(f"hyperparameter_search_ranges: {hyperparameter_search_ranges}")
    tprint(f"search_db_url: {search_db_url}")
    tprint(f"plots_directory: {plots_directory}")
    tprint(f"search_db: {args.search_db}")

    columns_to_use = [title_header, abstract_header, label_header]

    df = pd.read_csv(data_path, sep="\t", usecols=columns_to_use)

    # First create a model
    print("")
    tprint(f"GETTING FEATURES")
    word_features = get_features(
        df,
        title_key=title_header,
        abstract_key=abstract_header,
        num_processes=1,
    )
    print("")
    tprint("GETTING LABELS")
    labels = get_labels(
        df, label_key=label_header, positive_class_value=positive_class_value
    )

    print("")
    tprint("GETTING TFIDF AND NAMES")
    tfidf_scores, feature_names = get_tfidf_and_names(word_features)

    print("")
    tprint("INITIALISING OPTIMISER")
    optimiser = OptunaHyperparameterOptimisation(
        tfidf_scores=tfidf_scores,
        labels=labels,
        model_name=model_name,
        max_n_search_iterations=max_n_search_iterations,
        n_jobs=-1,
        nfolds=nfolds,
        num_cv_repeats=num_cv_repeats,
        db_url=search_db_url,
        user_selected_hyperparameter_ranges=hyperparameter_search_ranges,
        timeout=timeout,
        use_early_terminator=use_early_terminator,
        max_stagnation_iterations=max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=use_worse_than_first_two_pruner,
    )

    optimiser.delete_optuna_study(study_name="hyperparameter_search_study")

    ("")
    tprint("RUNNING HYPERPARAMETER SEARCH")
    study = optimiser.run_hparam_search_study(study_name="hyperparameter_search_study")

    print("")
    tprint("GET BEST HYPERPARAMETERS FROM STUDY")
    best_params = get_best_hyperparams_from_study(study)

    # Second perform analysis for the resulting model

    optuna_plots_directory = f"{plots_directory}/optuna_plots"
    create_all_optuna_plots(study, optuna_plots_directory)

    fold_scores, fold_labels = predict_cv_scores(
        tfidf_scores, labels, model_name, best_params, nfolds=nfolds, num_cv_repeats=1
    )

    cv_y_scores = np.concatenate(fold_scores, axis=0)
    cv_y = np.concatenate(fold_labels, axis=0)

    roc_plot(cv_y, cv_y_scores, f"{plots_directory}/roc_plot.html")

    select_threshold_plot(
        cv_y, cv_y_scores, f"{plots_directory}/select_threshold_plot.html"
    )

    positive_negative_scores_histogram_plot(
        cv_y,
        cv_y_scores,
        f"{plots_directory}/positive_negative_scores_histogram_plot.html",
    )

    box_plot(
        data_by_box=fold_scores,
        box_names=[f"Fold {i}" for i in range(nfolds)],
        title="Cross-Validation AUC Scores",
        yaxis_title="AUC",
        xaxis_title="",
        savepath=f"{plots_directory}/box_plot.html",
    )

    learning_curve(
        tfidf_scores,
        labels,
        model_name,
        best_params,
        savepath=f"{plots_directory}/learning_curve.html",
        nfolds=nfolds,
        proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    auc_scores = predict_cv_metrics(
        tfidf_scores=tfidf_scores,
        labels=labels,
        model_name=model_name,
        model_params=best_params,
        nfolds=3,
        num_cv_repeats=100,
    )

    histogram_plot(
        scores=auc_scores,
        savepath=f"{plots_directory}/auc_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="AUC",
    )

    model = train(
        model_name=model_name,
        params=best_params,
        X=tfidf_scores,
        y=labels,
        n_jobs=-1,
    )
    shap_plotter = ShapPlotter(
        model,
        tfidf_scores,
        feature_names,
    )
    dot_plot = shap_plotter.dot_plot(num_display=shap_num_display, log_scale=True)
    dot_plot.show()
    dot_plot = shap_plotter.dot_plot(num_display=shap_num_display, log_scale=False)
    dot_plot.show()
    bar_plot = shap_plotter.bar_chart(num_display=shap_num_display)
    bar_plot.show()

    save_npz(f"{args.tfidf_scores}/tfidf.npz", tfidf_scores)
    np.save(f"{args.feature_names}/feature_names.npy", feature_names)
    np.save(f"{args.labels}/lables.npy", labels)
    with open(f"{args.best_hparams}/best_hparams.json", "w") as f:
        json.dump(best_params, f)


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    main()
