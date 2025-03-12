import argparse
from datetime import datetime
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


@dataclass(config=ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True))
class ThresholdModelAnalysisArgs:
    # Inputs
    labelled_tfidf_scores: csr_matrix
    labels: np.ndarray
    model_name: Literal["lightgbm", "xgboost", "RandomForestClassifier", "SVC"]
    model_params: dict
    threshold: float
    nfolds: Annotated[int, Field(gt=2, le=10)]
    histogram_num_cv_repeats: Annotated[int, Field(gt=0, le=500)]
    confusion_num_cv_repeats: Annotated[int, Field(gt=0, le=10)]

    # Outputs
    plots_dir: str


def parse_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelled_tfidf_scores",
        type=str,
        help="Path to directory of tfidf_scores",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="Path to directory of labels",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    parser.add_argument(
        "--model_params",
        type=str,
        help="Path to directory of the model parameters",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold for positive class",
    )
    parser.add_argument(
        "--nfolds",
        type=int,
        help="Number of cross validation folds",
        default=3,
    )
    parser.add_argument(
        "--histogram_num_cv_repeats",
        type=int,
        help="Number of cv repeats to do for histogram plot",
        default=100,
    )
    parser.add_argument(
        "--confusion_num_cv_repeats",
        type=int,
        help="Number of cv repeats to do for confusion plot",
        default=1,
    )
    parser.add_argument(
        "--plots",
        type=str,
        help="Path to directory to save plots",
    )
    args = parser.parse_args()
    print(args.model_params)
    return ThresholdModelAnalysisArgs(
        labelled_tfidf_scores=load_csr_at_directory(args.labelled_tfidf_scores),
        labels=load_np_array_at_directory(args.labels),
        model_name=args.model_name,
        model_params=load_json_at_directory(args.model_params),
        threshold=args.threshold,
        nfolds=args.nfolds,
        histogram_num_cv_repeats=args.histogram_num_cv_repeats,
        confusion_num_cv_repeats=args.confusion_num_cv_repeats,
        plots_dir=args.plots,
    )


def main(args: ThresholdModelAnalysisArgs):
    # Print all the arguments using the args dataclass
    tprint(f"labelled_tfidf_scores shape: {args.labelled_tfidf_scores.shape}")
    tprint(f"labelled_tfidf_scores nnz: {args.labelled_tfidf_scores.nnz}")
    tprint(f"labels.shape: {args.labels.shape}")
    tprint(f"model_name: {args.model_name}")
    tprint(f"model_params: {args.model_params}")
    tprint(f"threshold: {args.threshold}")
    tprint(f"nfolds: {args.nfolds}")
    tprint(f"histogram_num_cv_repeats: {args.histogram_num_cv_repeats}")
    tprint(f"confusion_cv_repeats: {args.confusion_num_cv_repeats}")
    tprint(f"plots_dir: {args.plots_dir}")
    print("")

    tprint(f"type of tfidf_scores: {type(args.labelled_tfidf_scores)}")
    tprint(f"type of labels: {type(args.labels)}")
    tprint(f"type of model_name: {type(args.model_name)}")
    tprint(f"type of model_params: {type(args.model_params)}")
    tprint(f"type of threshold: {type(args.threshold)}")
    tprint(f"type of nfolds: {type(args.nfolds)}")
    tprint(f"type of histogram_num_cv_repeats: {type(args.histogram_num_cv_repeats)}")
    tprint(f"type of confusion_num_cv_repeats: {type(args.confusion_num_cv_repeats)}")
    tprint(f"type of plots_dir: {type(args.plots_dir)}")
    print("")

    auc_scores, recall_scores, fpr_scores = predict_cv_metrics(
        tfidf_scores=args.labelled_tfidf_scores,
        labels=args.labels,
        model_name=args.model_name,
        model_params=args.model_params,
        nfolds=args.nfolds,
        num_cv_repeats=args.histogram_num_cv_repeats,
        threshold=args.threshold,
    )

    histogram_plot(
        scores=auc_scores,
        savepath=f"{args.plots_dir}/auc_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="AUC",
    )
    histogram_plot(
        recall_scores,
        f"{args.plots_dir}/recall_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="Recall",
        colour="rgba(50, 205, 50, 0.7)",
    )
    histogram_plot(
        fpr_scores,
        f"{args.plots_dir}/fpr_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="False Positive Rate",
        colour="rgb(251, 173, 60, 0.7)",
    )

    # Predict cv scores for confusion plot
    train_fold_scores, train_fold_labels, val_fold_scores, val_fold_labels = (
        predict_cv_scores(
            tfidf_scores=args.labelled_tfidf_scores,
            labels=args.labels,
            model_name=args.model_name,
            model_params=args.model_params,
            nfolds=args.nfolds,
            num_cv_repeats=args.confusion_num_cv_repeats,
        )
    )

    cv_train_scores = np.concatenate(train_fold_scores, axis=0)
    cv_train_pred = (cv_train_scores >= args.threshold).astype(int)
    cv_train_y = np.concatenate(train_fold_labels, axis=0)

    cv_val_scores = np.concatenate(val_fold_scores, axis=0)
    cv_val_pred = (cv_val_scores >= args.threshold).astype(int)
    cv_val_y = np.concatenate(val_fold_labels, axis=0)

    binary_train_valid_confusion_plotly(
        y_train=cv_train_y,
        y_train_pred=cv_train_pred,
        y_val=cv_val_y,
        y_val_pred=cv_val_pred,
        save_path=f"{args.plots_dir}/confusion_plot.html",
    )


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
