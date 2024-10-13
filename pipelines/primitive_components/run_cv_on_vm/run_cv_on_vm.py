import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from eppi_text_classification.utils import load_value_from_json_at_directory


def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="path to input dataframe")
    parser.add_argument("--tfidf_scores", type=str, help="path to ordered list of labels")
    parser.add_argument(
        "--labels", type=str, help="path to tfidf scores for data"
    )
    parser.add_argument(
        "--params", type=str, help="path to ordered list of feature names"
    )
    parser.add_argument(
        "--num_cv_repeats", type=str, help="path to name of title columns",
    )
    parser.add_argument(
        "--n_folds", type=str, help="path to name of abstract column",
    )
    args = parser.parse_args()

    title_header = args.title_header
    abstract_header = args.abstract_header
    columns_to_use = [title_header, abstract_header]



    np.save(os.path.join(args.scores, "scores.npy"), scores)
    save_npz(os.path.join(args.tfidf_scores, "tfidf_scores.npz"), tfidf_scores)


def run_cv(model, tfidf_scores, labels, params, num_cv_repeats, nfolds):
    scores = []
    for i in range(num_cv_repeats):
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
            X_train = tfidf_scores[train_idx]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx]
            y_val = labels[val_idx]

            model.fit(X_train, y_train)

            y_val_pred = _predict_scores(model, X_val)

            auc = roc_auc_score(y_val, y_val_pred)
            scores.append(auc)

            # Prune if need to
    #                if self.use_pruner:
    #                    should_prune = self.should_we_prune(trial, scores)
    #                    if should_prune:
    #                        print(f"Pruned trial with scores: {scores}")
    #                        return np.mean(scores)

    return np.mean(scores)

if __name__ == "__main__":
    main()
