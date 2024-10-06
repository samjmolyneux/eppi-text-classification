
import argparse
import json
import os

from eppi_text_classification import get_raw_threshold
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_joblib_model_at_directory,
    load_np_array_at_directory,
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
        "--y",
        type=str,
        help="path to y data",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="path to model",
    )
    parser.add_argument(
        "--target_tpr",
        type=float,
        help="target true positive rate",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        help="path to threshold",
    )
    args = parser.parse_args()

    model = load_joblib_model_at_directory(args.model)
    X = load_csr_at_directory(args.X)
    y = load_np_array_at_directory(args.y)
    target_tpr = args.target_tpr

    threshold = get_raw_threshold(model, X, y, target_tpr)

    print(f"threshold: {threshold}")
    with open(os.path.join(args.threshold, "threshold.json"), "w") as file:
        json.dump(threshold, file)


if __name__ == "__main__":
    main()
