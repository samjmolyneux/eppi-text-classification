import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.sparse import save_npz

from eppi_text_classification import get_features, get_labels, get_tfidf_and_names
from eppi_text_classification.model_io import save_model_to_dir
from eppi_text_classification.model_stability import (
    predict_cv_metrics,
    predict_cv_scores,
)
from eppi_text_classification.multi_process_hparam_search import (
    MultiProcessHparamSearch,
    get_best_hparams_from_study,
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
from eppi_text_classification.plots.shap_plotter import ShapPlotter
from eppi_text_classification.train import train
from eppi_text_classification.utils import load_json_at_directory, str2bool


@dataclass(config=ConfigDict(frozen=True, strict=True))
class SingleModelArgs:
    # Inputs
    labelled_data_path: str
    unlabelled_data_path: str
    title_header: Annotated[str, Field(min_length=1)]
    abstract_header: Annotated[str, Field(min_length=1)]
    label_header: Annotated[str, Field(min_length=1)]
    positive_class_value: Annotated[str, Field(min_length=1)]
    model_name: Literal["lightgbm", "xgboost", "RandomForestClassifier", "SVC"]
    hparam_search_ranges: dict
    max_n_search_iterations: Annotated[int, Field(gt=0)] | None
    nfolds: Annotated[int, Field(ge=3, le=10)]
    num_cv_repeats: Annotated[int, Field(gt=0)]
    timeout: Annotated[int, Field(gt=0, le=3600 * 24)]
    use_early_terminator: bool
    max_stagnation_iterations: Annotated[int, Field(gt=25)] | None
    wilcoxon_trial_pruner_threshold: Annotated[float, Field(gt=0, le=1)] | None
    use_worse_than_first_two_pruner: bool
    shap_num_display: Annotated[int, Field(ge=10)]

    # Outputs
    search_db_dir: str
    feature_names_dir: str
    labelled_tfidf_dir: str
    unlabelled_tfidf_dir: str
    labels_dir: str
    plots_dir: str
    best_hparams_dir: str
    trained_model_dir: str

    # TODO validate the hparam_search_ranges


def parse_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelled_data",
        type=str,
        help="path to input tsv",
    )
    parser.add_argument(
        "--unlabelled_data",
        type=str,
        help="path to input tsv",
    )
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
        "--hparam_search_ranges",
        type=str,
        help="Path to directory containing hyperparameter search ranges json",
        default=None,
    )
    parser.add_argument(
        "--max_n_search_iterations",
        type=int,
        help="Maximum number of search iterations",
        default=None,
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
        default=3600 * 24,
    )
    parser.add_argument(
        "--use_early_terminator",
        type=str2bool,
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
        type=str2bool,
        help="Whether to use the worse than first two pruner",
        default=False,
    )
    parser.add_argument(
        "--shap_num_display",
        type=int,
        help="Number of features to display in shap plots",
        default=20,
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
        "--labelled_tfidf_scores",
        type=str,
        help="Path to tfidf scores",
    )
    parser.add_argument(
        "--unlabelled_tfidf_scores",
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
    parser.add_argument(
        "--trained_model",
        type=str,
        help="Path directory of the model",
    )
    args = parser.parse_args()

    return SingleModelArgs(
        labelled_data_path=args.labelled_data,
        unlabelled_data_path=args.unlabelled_data,
        title_header=args.title_header,
        abstract_header=args.abstract_header,
        label_header=args.label_header,
        positive_class_value=args.positive_class_value,
        model_name=args.model_name,
        hparam_search_ranges=load_json_at_directory(args.hparam_search_ranges),
        max_n_search_iterations=args.max_n_search_iterations,
        nfolds=args.nfolds,
        num_cv_repeats=args.num_cv_repeats,
        timeout=args.timeout,
        use_early_terminator=args.use_early_terminator,
        max_stagnation_iterations=args.max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=args.wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=args.use_worse_than_first_two_pruner,
        shap_num_display=args.shap_num_display,
        search_db_dir=args.search_db,
        feature_names_dir=args.feature_names,
        labelled_tfidf_dir=args.labelled_tfidf_scores,
        unlabelled_tfidf_dir=args.unlabelled_tfidf_scores,
        labels_dir=args.labels,
        plots_dir=args.plots,
        best_hparams_dir=args.best_hparams,
        trained_model_dir=args.trained_model,
    )


def main(args: SingleModelArgs):
    print(f"pwd: {Path.cwd()}")
    print("")

    search_db_url = f"sqlite:///{args.search_db_dir}optuna.db"
    # search_db_url = f"sqlite:///{cwd_path.parents[5]}/optuna.db"
    print(f"search_db_url: {search_db_url}")

    os.mkdir(f"{args.plots_dir}/optuna_plots")

    # raise Exception("stop")

    # Print all the arguments using the args labelled_dataclass
    tprint(f"labelled_data_path: {args.labelled_data_path}")
    tprint(f"unlabelled_data_path: {args.unlabelled_data_path}")
    tprint(f"title_header: {args.title_header}")
    tprint(f"abstract_header: {args.abstract_header}")
    tprint(f"label_header: {args.label_header}")
    tprint(f"positive_class_value: {args.positive_class_value}")
    tprint(f"model_name: {args.model_name}")
    tprint(f"max_n_search_iterations: {args.max_n_search_iterations}")
    tprint(f"nfolds: {args.nfolds}")
    tprint(f"num_cv_repeats: {args.num_cv_repeats}")
    tprint(f"timeout: {args.timeout}")
    tprint(f"use_early_terminator: {args.use_early_terminator}")
    tprint(f"max_stagnation_iterations: {args.max_stagnation_iterations}")
    tprint(f"wilcoxon_trial_pruner_threshold: {args.wilcoxon_trial_pruner_threshold}")
    tprint(f"use_worse_than_first_two_pruner: {args.use_worse_than_first_two_pruner}")
    tprint(f"shap_num_display: {args.shap_num_display}")
    tprint(f"hyperparameter_search_ranges: {args.hparam_search_ranges}")
    tprint(f"search_db_url: {search_db_url}")
    tprint(f"plots_dir: {args.plots_dir}")
    tprint(f"search_db: {args.search_db_dir}")
    tprint(f"labelled_tfidf_dir: {args.labelled_tfidf_dir}")
    tprint(f"unlabelled_tfidf_dir: {args.unlabelled_tfidf_dir}")
    tprint(f"trained_model_dir: {args.trained_model_dir}")

    # Print types of all the arguments
    tprint(f"type of labelled_data_path: {type(args.labelled_data_path)}")
    tprint(f"type of unlabelled_data_path: {type(args.unlabelled_data_path)}")
    tprint(f"type of title_header: {type(args.title_header)}")
    tprint(f"type of abstract_header: {type(args.abstract_header)}")
    tprint(f"type of label_header: {type(args.label_header)}")
    tprint(f"type of positive_class_value: {type(args.positive_class_value)}")
    tprint(f"type of model_name: {type(args.model_name)}")
    tprint(f"type of max_n_search_iterations: {type(args.max_n_search_iterations)}")
    tprint(f"type of nfolds: {type(args.nfolds)}")
    tprint(f"type of num_cv_repeats: {type(args.num_cv_repeats)}")
    tprint(f"type of timeout: {type(args.timeout)}")
    tprint(f"type of use_early_terminator: {type(args.use_early_terminator)}")
    tprint(f"type of max_stagnation_iterations: {type(args.max_stagnation_iterations)}")
    tprint(
        f"type of wilcoxon_trial_pruner_threshold: {type(args.wilcoxon_trial_pruner_threshold)}"
    )
    tprint(
        f"type of use_worse_than_first_two_pruner: {type(args.use_worse_than_first_two_pruner)}"
    )
    tprint(f"type of shap_num_display: {type(args.shap_num_display)}")
    tprint(f"type of hyperparameter_search_ranges: {type(args.hparam_search_ranges)}")
    tprint(f"type of search_db_url: {type(search_db_url)}")
    tprint(f"type of plots_dir: {type(args.plots_dir)}")
    tprint(f"type of search_db: {type(args.search_db_dir)}")
    tprint(f"type labelled_tfidf_dir: {type(args.labelled_tfidf_dir)}")
    tprint(f"type unlabelled_tfidf_dir: {type(args.unlabelled_tfidf_dir)}")
    tprint(f"type of trained_model_dir: {type(args.trained_model_dir)}")

    labelled_df = pd.read_csv(
        args.labelled_data_path,
        sep="\t",
        usecols=[args.title_header, args.abstract_header, args.label_header],
    )

    unlabelled_df = pd.read_csv(
        args.unlabelled_data_path,
        sep="\t",
        usecols=[args.title_header, args.abstract_header],
    )

    # First create a model
    print("")
    tprint(f"GETTING LABELLED WORD FEATURES")
    labelled_word_features = get_features(
        labelled_df,
        title_key=args.title_header,
        abstract_key=args.abstract_header,
    )

    print("")
    tprint(f"GETTING UNLABELLED WORD FEATURES")
    unlabelled_word_features = get_features(
        unlabelled_df,
        title_key=args.title_header,
        abstract_key=args.abstract_header,
    )

    word_features = labelled_word_features + unlabelled_word_features

    # Check that nothing was lost in word feature extraction
    number_of_rows = labelled_df.shape[0] + unlabelled_df.shape[0]
    assert number_of_rows == len(
        word_features
    ), "something was lost in word feature extraction"

    print("")
    tprint("GETTING LABELS")
    labels = get_labels(
        labelled_df,
        label_key=args.label_header,
        positive_class_value=args.positive_class_value,
    )

    print("")
    tprint("GETTING TFIDF AND NAMES")
    all_tfidf_scores, feature_names = get_tfidf_and_names(word_features)

    labelled_tfidf_scores = all_tfidf_scores[: labelled_df.shape[0]]
    unlabelled_tfidf_scores = all_tfidf_scores[labelled_df.shape[0] :]

    print("")
    tprint("INITIALISING OPTIMISER")
    optimiser = MultiProcessHparamSearch(
        tfidf_scores=labelled_tfidf_scores,
        labels=labels,
        model_name=args.model_name,
        max_n_search_iterations=args.max_n_search_iterations,
        n_jobs=-1,
        nfolds=args.nfolds,
        num_cv_repeats=args.num_cv_repeats,
        db_url=search_db_url,
        user_selected_hyperparameter_ranges=args.hparam_search_ranges,
        timeout=args.timeout,
        use_early_terminator=args.use_early_terminator,
        max_stagnation_iterations=args.max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=args.wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=args.use_worse_than_first_two_pruner,
    )

    optimiser.delete_optuna_study(study_name="hyperparameter_search_study")

    print("")
    tprint("RUNNING HYPERPARAMETER SEARCH")
    study = optimiser.run_hparam_search_study(study_name="hyperparameter_search_study")

    print("")
    tprint("GET BEST HYPERPARAMETERS FROM STUDY")
    best_params = get_best_hparams_from_study(study)

    # Second perform analysis for the resulting model

    optuna_plots_directory = f"{args.plots_dir}/optuna_plots"
    create_all_optuna_plots(study, optuna_plots_directory)

    _, _, val_fold_scores, val_fold_labels = predict_cv_scores(
        labelled_tfidf_scores,
        labels,
        args.model_name,
        best_params,
        nfolds=args.nfolds,
        num_cv_repeats=1,
    )

    cv_scores = np.concatenate(val_fold_scores, axis=0)
    cv_y = np.concatenate(val_fold_labels, axis=0)

    roc_plot(cv_y, cv_scores, f"{args.plots_dir}/roc_plot.html")

    select_threshold_plot(
        cv_y, cv_scores, f"{args.plots_dir}/select_threshold_plot.html"
    )

    positive_negative_scores_histogram_plot(
        cv_y,
        cv_scores,
        f"{args.plots_dir}/positive_negative_scores_histogram_plot.html",
    )

    box_plot(
        data_by_box=val_fold_scores,
        box_names=[f"Fold {i}" for i in range(args.nfolds)],
        title="Cross-Validation AUC Scores",
        yaxis_title="AUC",
        xaxis_title="",
        savepath=f"{args.plots_dir}/box_plot.html",
    )

    learning_curve(
        labelled_tfidf_scores,
        labels,
        args.model_name,
        best_params,
        savepath=f"{args.plots_dir}/learning_curve.html",
        nfolds=args.nfolds,
        proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )

    auc_scores = predict_cv_metrics(
        tfidf_scores=labelled_tfidf_scores,
        labels=labels,
        model_name=args.model_name,
        model_params=best_params,
        nfolds=3,
        num_cv_repeats=100,
    )

    histogram_plot(
        scores=auc_scores,
        savepath=f"{args.plots_dir}/auc_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="AUC",
    )

    model = train(
        model_name=args.model_name,
        params=best_params,
        X=labelled_tfidf_scores,
        y=labels,
        n_jobs=-1,
    )
    shap_plotter = ShapPlotter(
        model,
        unlabelled_tfidf_scores,
        feature_names,
    )
    dot_plot = shap_plotter.dot_plot(num_display=args.shap_num_display, log_scale=True)
    dot_plot.save(filename=f"{args.plots_dir}/shap_dot_plot_log_x.png")
    dot_plot = shap_plotter.dot_plot(num_display=args.shap_num_display, log_scale=False)
    dot_plot.save(filename=f"{args.plots_dir}/shap_dot_plot_linear_x.png")
    bar_plot = shap_plotter.bar_chart(num_display=args.shap_num_display)
    bar_plot.save(filename=f"{args.plots_dir}/shap_bar_plot.png")

    save_npz(f"{args.labelled_tfidf_dir}/labelled_tfidf.npz", labelled_tfidf_scores)
    save_npz(
        f"{args.unlabelled_tfidf_dir}/unlabelled_tfidf.npz", unlabelled_tfidf_scores
    )
    np.save(f"{args.feature_names_dir}/feature_names.npy", feature_names)
    np.save(f"{args.labels_dir}/labels.npy", labels)
    with open(f"{args.best_hparams_dir}/best_hparams.json", "w") as f:
        json.dump(best_params, f)
    print(type(model))
    save_model_to_dir(model, args.trained_model_dir)


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
