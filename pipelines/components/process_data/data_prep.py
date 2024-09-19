import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from eppi_text_classification import (
    get_features_and_labels,
    get_tfidf_and_names,
)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input dataframe")
    parser.add_argument("--labels", type=str, help="path to ordered list of labels")
    parser.add_argument(
        "--tfidf_scores", type=str, help="path to tfidf scores for data"
    )
    parser.add_argument(
        "--feature_names", type=str, help="path to ordered list of feature names"
    )
    args = parser.parse_args()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    df = pd.read_csv(args.data, sep="\t")

    word_features, labels = get_features_and_labels(df)
    tfidf_scores, feature_names = get_tfidf_and_names(word_features)

    print(f"labels: {args.labels}")
    print(f"feature_names: {args.feature_names}")
    print(f"tfidf_scores: {args.tfidf_scores}")

    np.save(os.path.join(args.labels, "labels.npy"), labels)
    np.save(os.path.join(args.feature_names, "feature_names.npy"), feature_names)
    save_npz(os.path.join(args.tfidf_scores, "tfidf_scores.npz"), tfidf_scores)


if __name__ == "__main__":
    main()
