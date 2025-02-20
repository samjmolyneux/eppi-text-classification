from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .predict import predict_scores, raw_threshold_predict
from .train import train


def predict_cv_metrics(
    tfidf_scores,
    labels,
    model_name,
    model_params,
    nfolds,
    num_cv_repeats,
    threshold=None,
):
    auc_scores = []
    recall_scores = []
    specifcity_scores = []

    for i in range(num_cv_repeats):
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)
        for _, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
            X_train = tfidf_scores[train_idx]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx]
            y_val = labels[val_idx]

            clf = train(model_name, model_params, X_train, y_train)

            y_val_scores = predict_scores(clf, X_val)

            auc = roc_auc_score(y_val, y_val_scores)
            auc_scores.append(auc)

            if threshold is not None:
                y_val_pred = raw_threshold_predict(clf, X_val, threshold)
                recall = recall_score(y_val, y_val_pred)
                recall_scores.append(recall)

                specifcity = recall_score(y_val, y_val_pred, pos_label=0)
                specifcity_scores.append(specifcity)

    if threshold is None:
        return auc_scores
    else:
        return auc_scores, recall_scores, specifcity_scores


def predict_cv_scores(
    tfidf_scores, labels, model_name, model_params, nfolds, num_cv_repeats
):
    fold_raw_scores = []

    for i in range(num_cv_repeats):
        kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)
        for _, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
            X_train = tfidf_scores[train_idx]
            X_val = tfidf_scores[val_idx]

            y_train = labels[train_idx]

            clf = train(model_name, model_params, X_train, y_train)

            y_val_scores = predict_scores(clf, X_val)
            fold_raw_scores.append(y_val_scores)

    return fold_raw_scores


def predict_cv_metrics_per_model(
    tfidf_scores,
    labels,
    model_names,
    model_params_list,
    nfolds,
    num_cv_repeats,
    thresholds=None,
):
    auc_scores_per_model = []
    recall_scores_per_model = []
    specificity_scores_per_model = []

    assert len(model_names) == len(
        model_params_list
    ), "model_list and model_param_list must have the same length"
    if thresholds is not None:
        assert len(model_names) == len(
            thresholds
        ), "model_list and thresholds must have the same length"

    for i in range(len(model_names)):
        model_name = model_names[i]
        model_params = model_params_list[i]
        if thresholds is not None:
            threshold = thresholds[i]

        auc_scores = []
        recall_scores = []
        specifcity_scores = []
        for i in range(num_cv_repeats):
            kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=i)

            for _, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
                X_train = tfidf_scores[train_idx]
                X_val = tfidf_scores[val_idx]

                y_train = labels[train_idx]
                y_val = labels[val_idx]

                clf = train(model_name, model_params, X_train, y_train)

                y_val_scores = predict_scores(clf, X_val)

                auc = roc_auc_score(y_val, y_val_scores)
                auc_scores.append(auc)

                if thresholds is not None:
                    y_val_pred = raw_threshold_predict(clf, X_val, threshold)
                    recall = recall_score(y_val, y_val_pred)
                    recall_scores.append(recall)

                    specifcity = recall_score(y_val, y_val_pred, pos_label=0)
                    specifcity_scores.append(specifcity)

        auc_scores_per_model.append(auc_scores)
        recall_scores_per_model.append(recall_scores)
        specificity_scores_per_model.append(specifcity_scores)

    if thresholds is None:
        return auc_scores_per_model
    else:
        return (
            auc_scores_per_model,
            recall_scores_per_model,
            specificity_scores_per_model,
        )
