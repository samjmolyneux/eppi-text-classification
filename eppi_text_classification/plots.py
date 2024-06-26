"""Basic matplotlib plots. For if you don't want to use plotly."""

from collections.abc import Sequence

from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def binary_train_valid_confusion_plot(
    y_train: Sequence,
    y_train_pred: Sequence,
    y_valid: Sequence,
    y_valid_pred: Sequence,
    positive_label: str = "1",
    negative_label: str = "0",
) -> None:
    """
    Plt plot of binary confusion matrix for training and validation data.

    Parameters
    ----------
    y_train : Sequence
        Truth labels for training data.

    y_train_pred : Sequence
        Predicted labels for training data.

    y_valid : Sequence
        Truth labels for validation data.

    y_valid_pred : Sequence
        Predicted labels for validation data.

    positive_label : str, optional
        Label for postive class. By default "1"

    negative_label : str, optional
        Label for negative class. By default "0"

    """
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_valid, y_valid_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=[negative_label, positive_label]
    )
    disp_train.plot(ax=ax[0], cmap=plt.cm.Blues)
    ax[0].set_title("Training Matrix")
    ax[0].set_xlabel("Prediction")
    ax[0].set_ylabel("Truth")

    disp_val = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=[negative_label, positive_label]
    )
    disp_val.plot(ax=ax[1], cmap=plt.cm.Blues)
    ax[1].set_title("Validation  Matrix")
    ax[1].set_xlabel("Prediction")
    ax[1].set_ylabel("Truth")

    plt.tight_layout()
    plt.show()
