import argparse
import json
from datetime import datetime
from typing import Literal

import numpy as np
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from eppi_text_classification.model_io import load_model_from_dir
from eppi_text_classification.predict import raw_threshold_predict
from eppi_text_classification.utils import (
    load_csr_at_directory,
    save_model_to_dir,
    save_npz,
)


@dataclass(config=ConfigDict(frozen=True, strict=True))
class PredArgs:
    # Inputs
    unlabelled_data_dir: str
    model_dir: str
    threshold: float

    # Outputs
    pred_labels_dir: str


def parse_args():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unlabelled_data",
        type=str,
        help="path to input tsv",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Classification threshold",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Path to saved model",
    )
    parser.add_argument(
        "--pred_labels",
        type=str,
        help="Predicted labels directory",
    )
    args = parser.parse_args()

    return PredArgs(
        unlabelled_data_dir=args.unlabelled_data,
        threshold=args.threshold,
        model_dir=args.model,
        pred_labels_dir=args.pred_labels,
    )


def main(args: PredArgs):
    # Print all the arguments using the args PredArgs dataclass
    tprint(f"unlabelled_data_dir: {args.unlabelled_data_dir}")
    tprint(f"model_dir: {args.model_dir}")
    tprint(f"threshold: {args.threshold}")
    tprint(f"pred_dir: {args.pred_labels_dir}")

    # Print types of all the arguments
    tprint(f"type of unlabelled_data_dir: {type(args.unlabelled_data_dir)}")
    tprint(f"type of threshold: {type(args.threshold)}")
    tprint(f"type of model_dir: {type(args.model_dir)}")
    tprint(f"type of pred_dir: {type(args.pred_labels_dir)}")

    model = load_model_from_dir(args.model_dir)

    unlabelled_tfidf_scores = load_csr_at_directory(args.unlabelled_data_dir)

    pred_labels = raw_threshold_predict(
        model=model,
        X=unlabelled_tfidf_scores,
        threshold=args.threshold,
    )

    np.save(f"{args.pred_labels_dir}/pred_labels.npy", pred_labels)


def tprint(*args, **kwargs):
    """Print with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}]", *args, **kwargs)


if __name__ == "__main__":
    args = parse_args()
    main(args)
