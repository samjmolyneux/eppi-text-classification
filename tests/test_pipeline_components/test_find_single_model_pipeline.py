import argparse
import datetime
import json
import os
import time
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
from pipelines.final_components import SingleModelArgs, run_find_single_model


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.datetime.now(datetime.UTC).strftime("%S-%M-%H-%d-%m-%Y")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    current_time = datetime.datetime.now(datetime.UTC).strftime("%S-%M-%H-%d-%m-%Y")
    paths = {
        "search_db_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/search_db",
        "feature_names_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/feature_names",
        "labelled_tfidf_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/labelled_tfidf",
        "unlabelled_tfidf_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/unlabelled_tfidf",
        "labels_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/labels",
        "plots_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/plots",
        "best_hparams_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/best_hparams",
        "trained_model_dir": f"sim_pipeline_outputs/find_single_model/{current_time}/trained_model",
    }
    args = SingleModelArgs(
        labelled_data_path="../../data/raw/debunking_review.tsv",
        unlabelled_data_path="../../data/raw/debunking_review.tsv",
        title_header="title",
        abstract_header="abstract",
        label_header="included",
        positive_class_value="1",
        # labelled_data_path="../../data/raw/hedges-all.tsv",
        # unlabelled_data_path="../../data/raw/hedges-all.tsv",
        # title_header="ti",
        # abstract_header="ab",
        # label_header="is_rct",
        # positive_class_value="1",
        model_name="SVC",
        # hparam_search_ranges=load_json_at_directory(
        #     "../../pipelines/user_inputs/uri_folders/hyperparameter_search_ranges/"
        hparam_search_ranges={},
        max_n_search_iterations=100,
        nfolds=4,
        num_cv_repeats=1,
        timeout=80000,
        use_early_terminator=False,
        max_stagnation_iterations=None,
        wilcoxon_trial_pruner_threshold=None,
        use_worse_than_first_two_pruner=False,
        shap_num_display=20,
        # Outputs
        search_db_dir=paths["search_db_dir"],
        feature_names_dir=paths["feature_names_dir"],
        labelled_tfidf_dir=paths["labelled_tfidf_dir"],
        unlabelled_tfidf_dir=paths["unlabelled_tfidf_dir"],
        labels_dir=paths["labels_dir"],
        plots_dir=paths["plots_dir"],
        best_hparams_dir=paths["best_hparams_dir"],
        trained_model_dir=paths["trained_model_dir"],
    )

    for path in paths.values():
        Path.mkdir(Path(path), parents=True)

    run_find_single_model(args)
