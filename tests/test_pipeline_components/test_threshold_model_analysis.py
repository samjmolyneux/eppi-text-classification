import argparse
import datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix, load_npz

from eppi_text_classification.model_stability import (
    predict_cv_metrics,
    predict_cv_scores,
)
from eppi_text_classification.plots import (
    binary_train_valid_confusion_plotly,
    histogram_plot,
)
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_json_at_directory,
    load_np_array_at_directory,
)
from pipelines.final_components import (
    ThresholdModelAnalysisArgs,
    run_threshold_model_analysis,
)

if __name__ == "__main__":
    dirs = [
        d
        for d in Path("./sim_pipeline_outputs/find_single_model/").iterdir()
        if d.is_dir()
    ]
    # Most recent outputs from single model is input dir
    input_dir = max(dirs, key=lambda d: d.stat().st_mtime)

    current_time = datetime.datetime.now(datetime.UTC).strftime("%S-%M-%H-%d-%m-%Y")
    plots_dir = f"./sim_pipeline_outputs/threshold_model_analysis/{current_time}/plots"

    args = ThresholdModelAnalysisArgs(
        labelled_tfidf_scores=load_csr_at_directory(input_dir / "labelled_tfidf"),
        labels=load_np_array_at_directory(input_dir / "labels"),
        model_name="lightgbm",
        model_params=load_json_at_directory(input_dir / "best_hparams"),
        threshold=-3.4,
        nfolds=4,
        histogram_num_cv_repeats=100,
        confusion_num_cv_repeats=1,
        plots_dir=plots_dir,
    )
    Path.mkdir(Path(plots_dir), parents=True, exist_ok=True)
    run_threshold_model_analysis(args)
