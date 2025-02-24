import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from eppi_text_classification import get_labels
from eppi_text_classification.utils import load_json_at_directory


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input dataframe")
    parser.add_argument(
        "--label_column_name",
        type=str,
        help="path to column name of the labels",
    )
    parser.add_argument(
        "--positive_class_value",
        type=str,
        help="path to the value of the positive class",
    )
    parser.add_argument("--labels", type=str, help="path to the labels")
    args = parser.parse_args()

    with open(args.label_column_name) as f:
        label_column_name = json.load(f)
    with open(args.positive_class_value) as f:
        positive_class_value = json.load(f)

    df = pd.read_csv(args.data, sep="\t", usecols=[label_column_name])

    labels = get_labels(
        df=df,
        label_column_name=label_column_name,
        positive_class_value=positive_class_value,
    )
    np.save(os.path.join(args.labels, "labels.npy"), labels)


if __name__ == "__main__":
    main()
