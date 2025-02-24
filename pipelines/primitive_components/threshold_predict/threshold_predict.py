
import argparse
import json
import os

import numpy as np

from eppi_text_classification import raw_threshold_predict
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_joblib_model_at_directory,
    load_value_from_json_at_directory,
)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--X",
        type=str,
        help="path to X data",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="path to model",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        help="path to threshold",
    )
    parser.add_argument(
        "--y_pred",
        type=str,
        help="path to y predictions",
    )
    args = parser.parse_args()

    model = load_joblib_model_at_directory(args.model)
    X = load_csr_at_directory(args.X)
    threshold = float(load_value_from_json_at_directory(args.threshold))

    y_pred = raw_threshold_predict(model, X, threshold)

    np.save(os.path.join(args.y_pred, "y_pred.npy"), y_pred)


if __name__ == "__main__":
    main()
