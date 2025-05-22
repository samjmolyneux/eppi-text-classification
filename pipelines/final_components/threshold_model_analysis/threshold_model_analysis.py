import argparse
import json
import os
from datetime import datetime
from typing import Annotated, Literal

import numpy as np
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
from numpy.typing import NDArray
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


@dataclass(config=ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True))
class ThresholdModelAnalysisArgs:
    # Inputs
    labelled_tfidf_scores: csr_matrix
    labels: NDArray[np.int_]
    model_name: Literal["lightgbm", "xgboost", "RandomForestClassifier", "SVC"]
    model_params: dict[str, float | int | str | bool]
    threshold: float
    nfolds: Annotated[int, Field(gt=2, le=10)]
    histogram_num_cv_repeats: Annotated[int, Field(gt=0, le=500)]
    confusion_num_cv_repeats: Annotated[int, Field(gt=0, le=10)]
    working_container_url: str
    output_container_path: str
    container_client: ContainerClient

    # Outputs
    # plots_dir: str


def parse_and_load_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labelled_tfidf_path",
        type=str,
        help="Path to directory of tfidf_scores",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        help="Path to directory of labels",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="Name of the model",
    )
    parser.add_argument(
        "--model_params_path",
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
        "--working_container_url",
        type=str,
        help="Container URL",
        default="https://eppidev2985087618.blob.core.windows.net/sams-prod-simulation",
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
        source_file_path=args.labelled_tfidf_path,
        destination_file_path="labelled_tfidf.npz",
    )

    download_blob_to_file(
        container_client=client,
        source_file_path=args.labels_path,
        destination_file_path="labels.npy",
    )
    labels = np.load("labels.npy")
    print(type(labels))

    download_blob_to_file(
        container_client=client,
        source_file_path=args.model_params_path,
        destination_file_path="model_params.json",
    )
    with open("model_params.json") as file:
        model_params = json.load(file)

    return ThresholdModelAnalysisArgs(
        labelled_tfidf_scores=load_npz("labelled_tfidf.npz"),
        labels=np.load("labels.npy"),
        model_name=args.model_name,
        model_params=model_params,
        threshold=args.threshold,
        nfolds=args.nfolds,
        histogram_num_cv_repeats=args.histogram_num_cv_repeats,
        confusion_num_cv_repeats=args.confusion_num_cv_repeats,
        working_container_url=args.working_container_url,
        output_container_path=args.output_container_path,
        container_client=client,
    )


def download_blob_to_file(
    container_client: ContainerClient,
    source_file_path: str,
    destination_file_path: str,
):
    with open(destination_file_path, "wb") as f:
        download_stream = container_client.download_blob(source_file_path)
        f.write(download_stream.readall())


def upload_blob_file(
    container_client: ContainerClient,
    source_file_path: str,
    destination_file_path: str,
):
    with open(file=source_file_path, mode="rb") as data:
        container_client.upload_blob(
            name=destination_file_path,
            data=data,
            overwrite=False,
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
    print("")

    tprint(f"type of tfidf_scores: {type(args.labelled_tfidf_scores)}")
    tprint(f"type of labels: {type(args.labels)}")
    tprint(f"type of model_name: {type(args.model_name)}")
    tprint(f"type of model_params: {type(args.model_params)}")
    tprint(f"type of threshold: {type(args.threshold)}")
    tprint(f"type of nfolds: {type(args.nfolds)}")
    tprint(f"type of histogram_num_cv_repeats: {type(args.histogram_num_cv_repeats)}")
    tprint(f"type of confusion_num_cv_repeats: {type(args.confusion_num_cv_repeats)}")
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

    os.makedirs("outputs/plots")

    # histogram_plot(
    #     scores=auc_scores,
    #     savepath="plots/auc_histogram.html",
    #     title="Model Stability Histogram",
    #     xaxis_title="AUC",
    # )
    histogram_plot(
        recall_scores,
        "outputs/plots/recall_histogram.html",
        title="Model Stability Histogram",
        xaxis_title="Recall",
        colour="rgba(50, 205, 50, 0.7)",
    )
    histogram_plot(
        fpr_scores,
        "outputs/plots/fpr_histogram.html",
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
        save_path="outputs/plots/confusion_plot.html",
    )

    # Upload all outputs to azure blob storage
    for path, _, file_names in os.walk("outputs"):
        for name in file_names:
            full_path = os.path.join(path, name)
            rel_path = os.path.relpath(full_path, start="outputs")
            upload_blob_file(
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
