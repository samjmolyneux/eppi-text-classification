from matplotlib import pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
)


def binary_train_valid_confusion_plot(
    y_train,
    y_train_pred,
    y_valid,
    y_valid_pred,
    positive_label="1",
    negative_label="0",
):
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_valid, y_valid_pred)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

    disp_train = ConfusionMatrixDisplay(
        confusion_matrix=cm_train, display_labels=[negative_label, positive_label]
    )
    disp_train.plot(ax=ax[0], cmap=plt.cm1.Blues)
    ax[0].set_title("Training Matrix")
    ax[0].set_xlabel("Prediction")
    ax[0].set_ylabel("Truth")

    disp_val = ConfusionMatrixDisplay(
        confusion_matrix=cm_test, display_labels=[negative_label, positive_label]
    )
    disp_val.plot(ax=ax[1], cmap=plt.cm1.Blues)
    ax[1].set_title("Validation  Matrix")
    ax[1].set_xlabel("Prediction")
    ax[1].set_ylabel("Truth")

    plt.tight_layout()
    plt.show()
