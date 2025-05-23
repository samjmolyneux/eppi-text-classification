import argparse
import os
from datetime import datetime

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import ContainerClient
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from scipy.sparse import csr_matrix, load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from eppi_text_classification.model_io import (
    load_model_from_filepath,
)
from eppi_text_classification.predict import raw_threshold_predict
from eppi_text_classification.utils import download_blob_to_file, upload_file_to_blob


@dataclass(config=ConfigDict(frozen=True, strict=True, arbitrary_types_allowed=True))
class PredArgs:
    # Inputs
    unlabelled_tfidf: csr_matrix
    model: lgb.basic.Booster | RandomForestClassifier | xgb.core.Booster | SVC
    threshold: float
    working_container_url: str
    output_container_path: str
    container_client: ContainerClient


def parse_and_load_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unlabelled_tfidf_path",
        type=str,
        help="path to input tsv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Classification threshold",
    )
    parser.add_argument(
        "--trained_model_dir",
        type=str,
        help="Path to saved model",
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
        source_file_path=args.unlabelled_tfidf_path,
        destination_file_path="unlabelled_tfidf.npz",
    )

    # The trained model directory contains only the model file, so we can
    # list all blobs in directory (just the model) and take the first one
    model_filepath = next(
        client.list_blob_names(name_starts_with=args.trained_model_dir)
    )
    download_blob_to_file(
        container_client=client,
        source_file_path=model_filepath,
        destination_file_path=os.path.basename(model_filepath),
    )
    return PredArgs(
        unlabelled_tfidf=load_npz("unlabelled_tfidf.npz"),
        threshold=args.threshold,
        model=load_model_from_filepath(os.path.basename(model_filepath)),
        working_container_url=args.working_container_url,
        output_container_path=args.output_container_path,
        container_client=client,
    )


def main(args: PredArgs):
    # Print all the arguments using the args PredArgs dataclass
    tprint(f"unlabelled_data: {args.unlabelled_tfidf.shape}")
    tprint(f"model: {args.model}")
    tprint(f"threshold: {args.threshold}")

    # Print types of all the arguments
    tprint(f"type of unlabelled_data_dir: {type(args.unlabelled_tfidf)}")
    tprint(f"type of threshold: {type(args.threshold)}")
    tprint(f"type of model_dir: {type(args.model)}")

    pred_labels = raw_threshold_predict(
        model=args.model,
        X=args.unlabelled_tfidf,
        threshold=args.threshold,
    )

    np.save("pred_labels.npy", pred_labels)
    upload_file_to_blob(
        container_client=args.container_client,
        source_file_path="pred_labels.npy",
        destination_file_path=f"{args.output_container_path}/pred_labels.npy",
    )


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    args = parse_and_load_args()
    main(args)
