"""Basic matplotlib plots. For if you don't want to use plotly."""

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def binary_train_valid_confusion_plot(
    y_train: ArrayLike,
    y_train_pred: ArrayLike,
    y_valid: ArrayLike,
    y_valid_pred: ArrayLike,
    positive_label: str = "1",
    negative_label: str = "0",
) -> None:
    """
    Plt plot of binary confusion matrix for training and validation data.

    Parameters
    ----------
    y_train : ArrayLike
        Truth labels for training data.

    y_train_pred : ArrayLike
        Predicted labels for training data.

    y_valid : ArrayLike
        Truth labels for validation data.

    y_valid_pred : ArrayLike
        Predicted labels for validation data.

    positive_label : str, optional
        Label for postive class. By default "1"

    negative_label : str, optional
        Label for negative class. By default "0"

    """
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_valid, y_valid_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300, squeeze=False)
    ax = np.squeeze(ax)
    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=[negative_label, positive_label]
    )
    disp_train.plot(ax=ax[0], cmap="Blues")
    ax[0].set_title("Training Matrix")
    ax[0].set_xlabel("Prediction")
    ax[0].set_ylabel("Truth")

    disp_val = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=[negative_label, positive_label]
    )
    disp_val.plot(ax=ax[1], cmap="Blues")
    ax[1].set_title("Validation  Matrix")
    ax[1].set_xlabel("Prediction")
    ax[1].set_ylabel("Truth")

    plt.tight_layout()
    plt.show(block=False)
