import argparse
import json
import os

import jsonpickle
import numpy as np
from scipy.sparse import load_npz, save_npz
from sklearn.model_selection import train_test_split

from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_np_array_at_directory,
)


def main():
    # input and output arguments
    print("before parse")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=str,
        help="path to ordered list of labels",
    )
    parser.add_argument(
        "--tfidf_scores",
        type=str,
        help="path to tfidf scores for data",
    )
    parser.add_argument(
        "--test_size",
        type=str,
        help="path to the test size as a proportion of the data",
    )
    parser.add_argument(
        "--X_train",
        type=str,
        help="path to X_train",
    )
    parser.add_argument(
        "--X_test",
        type=str,
        help="path to X_test",
    )
    parser.add_argument(
        "--y_train",
        type=str,
        help="path to y_train",
    )
    parser.add_argument(
        "--y_test",
        type=str,
        help="path to y_test",
    )
    args = parser.parse_args()
    tfidf_scores = load_csr_at_directory(args.tfidf_scores)
    labels = load_np_array_at_directory(args.labels)
    with open(args.test_size, "r") as file:
        test_size = float(json.load(file))

    print(f"test_size: {test_size}")
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_scores, labels, test_size=test_size, stratify=labels, random_state=8
    )

    save_npz(os.path.join(args.X_train, "X_train.npz"), X_train)
    save_npz(os.path.join(args.X_test, "X_test.npz"), X_test)
    np.save(os.path.join(args.y_train, "y_train.npy"), y_train)
    np.save(os.path.join(args.y_test, "y_test.npy"), y_test)


if __name__ == "__main__":
    main()
