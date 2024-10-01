
import argparse
import os

import numpy as np

from eppi_text_classification import predict_scores
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_joblib_model_at_directory,
)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--X",
        type=str,
        help="path to prediction data",
    )
    parser.add_argument(
        "--y_pred_probs",
        type=str,
        help="path to the predicted probabilities",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="path to trained model",
    )
    args = parser.parse_args()

    X = load_csr_at_directory(args.X)
    model = load_joblib_model_at_directory(args.model)

    y_pred_probabilities = predict_scores(model, X)

    np.save(os.path.join(args.y_pred_probs, "y_pred_probs.npy"), y_pred_probabilities)


if __name__ == "__main__":
    main()
