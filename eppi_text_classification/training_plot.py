from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .predict import predict_scores, raw_threshold_predict
from .train import train


def get_training_curve_data(
    tfidf_scores,
    labels,
    model_name,
    model_params,
    nfolds=5,
    proportions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
):
    train_sizes = [0]
    train_curve_scores = [0]
    val_curve_scores = [0]

    kf = StratifiedKFold(n_splits=nfolds, shuffle=True)
    for proportion in proportions:
        train_auc_scores = []
        val_auc_scores = []
        fold_train_sizes = []

        for _, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
            train_idx_slice = train_idx[: int(len(train_idx) * proportion)]

            fold_train_sizes.append(len(train_idx_slice))

            X_train = tfidf_scores[train_idx_slice]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx_slice]
            y_val = labels[val_idx]

            clf = train(model_name, model_params, X_train, y_train)

            y_train_scores = predict_scores(clf, X_train)
            y_val_scores = predict_scores(clf, X_val)

            train_auc = roc_auc_score(y_train, y_train_scores)
            train_auc_scores.append(train_auc)

            val_auc = roc_auc_score(y_val, y_val_scores)
            val_auc_scores.append(val_auc)

        train_sizes.append(np.mean(fold_train_sizes))
        train_curve_scores.append(np.mean(train_auc_scores))
        val_curve_scores.append(np.mean(val_auc_scores))

    return train_sizes, train_curve_scores, val_curve_scores
