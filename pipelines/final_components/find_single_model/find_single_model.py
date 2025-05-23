import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.storage.blob import ContainerClient
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
from eppi_text_classification.utils import (
    download_blob_to_file,
    str2bool,
    upload_file_to_blob,
)


@dataclass(config=ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True))
class SingleModelArgs:
    # Inputs
    labelled_df: pd.DataFrame
    unlabelled_df: pd.DataFrame
    title_header: Annotated[str, Field(min_length=1)]
    abstract_header: Annotated[str, Field(min_length=1)]
    label_header: Annotated[str, Field(min_length=1)]
    positive_class_value: Annotated[str, Field(min_length=1)]
    model_name: Literal["lightgbm", "xgboost", "RandomForestClassifier", "SVC"]
    hparam_search_ranges: (
        dict[str, dict[str, dict[str, float | int | str | bool]]] | None
    )
    max_n_search_iterations: Annotated[int, Field(gt=0)] | None
    nfolds: Annotated[int, Field(ge=3, le=10)]
    num_cv_repeats: Annotated[int, Field(gt=0)]
    timeout: Annotated[int, Field(gt=0, le=3600 * 24)]
    use_early_terminator: bool
    max_stagnation_iterations: Annotated[int, Field(gt=25)] | None
    wilcoxon_trial_pruner_threshold: Annotated[float, Field(gt=0, le=1)] | None
    use_worse_than_first_two_pruner: bool
    shap_num_display: Annotated[int, Field(ge=10)]
    working_container_url: str
    output_container_path: str
    container_client: ContainerClient


def parse_and_load_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelled_data_path",
        type=str,
        help="Path to labelled data",
    )
    parser.add_argument(
        "--unlabelled_data_path",
        type=str,
        help="Path to unlabelled data",
    )
    parser.add_argument(
        "--title_header",
        type=str,
        help="Name of title column",
    )
    parser.add_argument(
        "--abstract_header",
        type=str,
        help="Name of abstract column",
    )
    parser.add_argument(
        "--label_header",
        type=str,
        help="Name of label column",
    )
    parser.add_argument(
        "--positive_class_value",
        type=str,
        help="What value is given to the positive class in the dataset",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model used in the search",
    )
    parser.add_argument(
        "--hparam_search_ranges_path",
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
    )
    parser.add_argument(
        "--num_cv_repeats",
        type=int,
        help="Number of times to average the cross-validation to stabilize the search",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Time in seconds before cancelling search",
    )
    parser.add_argument(
        "--use_early_terminator",
        type=str2bool,
        help="Whether to use the regret based early terminator",
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
    )
    parser.add_argument(
        "--shap_num_display",
        type=int,
        help="Number of features to display in shap plots",
    )
    parser.add_argument(
        "--working_container_url",
        type=str,
        help="Container URL",
    )
    parser.add_argument(
        "--output_container_path",
        type=str,
        help="Path to output directory",
    )

    args = parser.parse_args()

    credential = ManagedIdentityCredential(
        client_id="df5b7af0-a55a-44d9-9ec7-9cde9abf3051"
    )
    client = ContainerClient.from_container_url(args.working_container_url, credential)

    download_blob_to_file(
        container_client=client,
        source_file_path=args.labelled_data_path,
        destination_file_path="labelled_data_path.tsv",
    )
    download_blob_to_file(
        container_client=client,
        source_file_path=args.unlabelled_data_path,
        destination_file_path="unlabelled_data_path.tsv",
    )

    labelled_df = pd.read_csv(
        "labelled_data_path.tsv",
        sep="\t",
        usecols=[args.title_header, args.abstract_header, args.label_header],
    )

    unlabelled_df = pd.read_csv(
        "unlabelled_data_path.tsv",
        sep="\t",
        usecols=[args.title_header, args.abstract_header],
    )

    if args.hparam_search_ranges_path is not None:
        download_blob_to_file(
            container_client=client,
            source_file_path=args.hparam_search_ranges_path,
            destination_file_path="hparam_search_ranges.json",
        )
        with open("hparam_search_ranges.json", "r") as f:
            hparam_search_ranges = json.load(f)
    else:
        hparam_search_ranges = None

    return SingleModelArgs(
        labelled_df=labelled_df,
        unlabelled_df=unlabelled_df,
        title_header=args.title_header,
        abstract_header=args.abstract_header,
        label_header=args.label_header,
        positive_class_value=args.positive_class_value,
        model_name=args.model_name,
        hparam_search_ranges=hparam_search_ranges,
        max_n_search_iterations=args.max_n_search_iterations,
        nfolds=args.nfolds,
        num_cv_repeats=args.num_cv_repeats,
        timeout=args.timeout,
        use_early_terminator=args.use_early_terminator,
        max_stagnation_iterations=args.max_stagnation_iterations,
        wilcoxon_trial_pruner_threshold=args.wilcoxon_trial_pruner_threshold,
        use_worse_than_first_two_pruner=args.use_worse_than_first_two_pruner,
        shap_num_display=args.shap_num_display,
        working_container_url=args.working_container_url,
        output_container_path=args.output_container_path,
        container_client=client,
    )


def main(args: SingleModelArgs):
    print(f"pwd: {Path.cwd()}")
    print("")

    optuna_db_dir = "outputs/optuna.db"
    search_db_url = f"sqlite:///{optuna_db_dir}"

    print(f"search_db_url: {search_db_url}")

    # Print all the arguments using the args labelled_dataclass
    tprint(f"labelled_data_path: {args.labelled_df.shape}")
    tprint(f"unlabelled_data_path: {args.unlabelled_df.shape}")
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
    tprint(f"working_container_url: {args.working_container_url}")
    tprint(f"output_container_path: {args.output_container_path}")

    # Print types of all the arguments
    tprint(f"type of labelled_data_path: {type(args.labelled_df)}")
    tprint(f"type of unlabelled_data_path: {type(args.unlabelled_df)}")
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
    tprint(f"type working_container_url: {type(args.working_container_url)}")
    tprint(f"type output_container_path: {type(args.output_container_path)}")

    # First create a model
    print("")
    tprint(f"GETTING LABELLED WORD FEATURES")
    labelled_word_features = get_features(
        args.labelled_df,
        title_key=args.title_header,
        abstract_key=args.abstract_header,
    )

    print("")
    tprint(f"GETTING UNLABELLED WORD FEATURES")
    unlabelled_word_features = get_features(
        args.unlabelled_df,
        title_key=args.title_header,
        abstract_key=args.abstract_header,
    )

    word_features = labelled_word_features + unlabelled_word_features

    # Check that nothing was lost in word feature extraction
    number_of_rows = args.labelled_df.shape[0] + args.unlabelled_df.shape[0]
    assert number_of_rows == len(
        word_features
    ), "something was lost in word feature extraction"

    print("")
    tprint("GETTING LABELS")
    labels = get_labels(
        args.labelled_df,
        label_key=args.label_header,
        positive_class_value=args.positive_class_value,
    )

    print("")
    tprint("GETTING TFIDF AND NAMES")
    all_tfidf_scores, feature_names = get_tfidf_and_names(word_features)

    labelled_tfidf_scores = all_tfidf_scores[: args.labelled_df.shape[0]]
    unlabelled_tfidf_scores = all_tfidf_scores[args.labelled_df.shape[0] :]

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

    optuna_plots_directory = "outputs/plots/optuna_plots"
    os.makedirs(optuna_plots_directory)
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

    roc_plot(cv_y, cv_scores, "outputs/plots/roc_plot.html")

    select_threshold_plot(cv_y, cv_scores, "outputs/plots/select_threshold_plot.html")

    positive_negative_scores_histogram_plot(
        cv_y,
        cv_scores,
        "outputs/plots/positive_negative_scores_histogram_plot.html",
    )

    box_plot(
        data_by_box=val_fold_scores,
        box_names=[f"Fold {i}" for i in range(args.nfolds)],
        title="Model Confidence Scores By Fold",
        yaxis_title="Scores",
        xaxis_title="",
        savepath="outputs/plots/box_plot.html",
    )

    learning_curve(
        labelled_tfidf_scores,
        labels,
        args.model_name,
        best_params,
        savepath="outputs/plots/learning_curve.html",
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
        savepath="outputs/plots/auc_histogram.html",
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
    dot_plot.save(filename="outputs/plots/shap_dot_plot_log_x.png")
    dot_plot = shap_plotter.dot_plot(num_display=args.shap_num_display, log_scale=False)
    dot_plot.save(filename="outputs/plots/shap_dot_plot_linear_x.png")
    bar_plot = shap_plotter.bar_chart(num_display=args.shap_num_display)
    bar_plot.save(filename="outputs/plots/shap_bar_plot.png")

    save_npz("outputs/labelled_tfidf.npz", labelled_tfidf_scores)
    save_npz("outputs/unlabelled_tfidf.npz", unlabelled_tfidf_scores)
    np.save("outputs/feature_names.npy", feature_names)
    np.save("outputs/labels.npy", labels)
    with open("outputs/best_hparams.json", "w") as f:
        json.dump(best_params, f)

    os.makedirs("outputs/trained_model")
    save_model_to_dir(model, "outputs/trained_model")

    # Upload all outputs to azure blob storage
    for path, _, file_names in os.walk("outputs"):
        for name in file_names:
            full_path = os.path.join(path, name)
            rel_path = os.path.relpath(full_path, start="outputs")
            upload_file_to_blob(
                container_client=args.container_client,
                source_file_path=full_path,
                destination_file_path=f"{args.output_container_path}/{rel_path}",
            )


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    args = parse_and_load_args()
    main(args)
