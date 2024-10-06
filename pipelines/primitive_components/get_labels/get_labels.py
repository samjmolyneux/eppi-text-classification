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
        "label_column_name",
        type=str,
        help="column name of the labels",
        default="included"
    )
    parser.add_argument(
        "positive_class_value",
        type=str,
        help="path to the value of the positive class",
        default="1"
    )
    parser.add_argument("--labels", type=str, help="path to the labels")
    args = parser.parse_args()

    label_column_name = args.label_column_name
    positive_class_value = parse_multiple_types(args.positive_class_value)

    df = pd.read_csv(args.data, sep="\t", usecols=[label_column_name])

    labels = get_labels(
        df=df,
        label_column_name=label_column_name,
        positive_class_value=positive_class_value,
    )
    np.save(os.path.join(args.labels, "labels.npy"), labels)

def parse_multiple_types(value):
    # Try to convert to int
    try:
        return int(value)
    except ValueError:
        pass
    
    # Try to convert to float
    try:
        return float(value)
    except ValueError:
        pass
    
    # Return as string if neither int nor float
    return value 

if __name__ == "__main__":
    main()
