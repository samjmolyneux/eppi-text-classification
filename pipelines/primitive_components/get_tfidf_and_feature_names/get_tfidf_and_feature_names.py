import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from eppi_text_classification import (
    get_features,
    get_tfidf_and_names,
)


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input dataframe")
    parser.add_argument(
        "--tfidf_scores", type=str, help="path to tfidf scores for data"
    )
    parser.add_argument(
        "--feature_names", type=str, help="path to ordered list of feature names"
    )
    parser.add_argument(
        "--title_header",
        type=str,
        help="path to name of title columns",
        default="title",
    )
    parser.add_argument(
        "--abstract_header",
        type=str,
        help="path to name of abstract column",
        default="abstract",
    )
    args = parser.parse_args()

    title_header = args.title_header
    abstract_header = args.abstract_header
    columns_to_use = [title_header, abstract_header]

    df = pd.read_csv(args.data, sep="\t", usecols=columns_to_use)

    word_features = get_features(df)
    tfidf_scores, feature_names = get_tfidf_and_names(word_features)

    np.save(os.path.join(args.feature_names, "feature_names.npy"), feature_names)
    save_npz(os.path.join(args.tfidf_scores, "tfidf_scores.npz"), tfidf_scores)


if __name__ == "__main__":
    main()
