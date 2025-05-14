"""Generate plotly confusion matrices. Plots can be saved to html."""

from typing import Literal

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from numpy.typing import ArrayLike, NDArray
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix,
)


def binary_train_valid_confusion_plotly(
    y_train: ArrayLike,
    y_train_pred: ArrayLike,
    y_val: ArrayLike,
    y_val_pred: ArrayLike,
    positive_label: str = "1",
    negative_label: str = "0",
    save_path: str | None = None,
) -> None:
    """
    Generate a binary classification confusion matrix for training and validation data.

    These plots can be saved to an html file.
    The plots are interactive in the browser.
    Change the output method at the bottom of the function.

    Parameters
    ----------
    y_train : ArrayLike
        The binary labels for the training data. (bool or int)

    y_train_pred : ArrayLike
        Predicted binary labels for the training data. (bool or int)

    y_val : ArrayLike
        The binary labels for the validation data. (bool or int)

    y_val_pred : ArrayLike
        _Predicted binary labels for the validation data. (bool or int)

    positive_label : str, optional
        The label for the positive class.
        Alters the pos label displayed when hovering over confusion matrix with cursor.
        By default "1".

    negative_label : str, optional
        The label for the negative class.
        Alters the neg label displayed when hovering over confusion matrix with cursor.
        By default "0".

    save_path : str, optional
        Path to save the plotly figure as an html file.

    """
    labels = ["0", "1"]
    cm1 = confusion_matrix(y_train, y_train_pred, labels=[0, 1])
    cm1 = np.array([[cm1[0, 1], cm1[0, 0]], [cm1[1, 1], cm1[1, 0]]])

    cm2 = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    cm2 = np.array([[cm2[0, 1], cm2[0, 0]], [cm2[1, 1], cm2[1, 0]]])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Training Matrix", "Validation Matrix"],
        horizontal_spacing=0.15,
    )

    # Increase font size and height of titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = {"size": 18}
        annotation["y"] = 1.03

    fig.update_layout(
        autosize=False,
        width=700,
        height=400,
    )

    add_confusion_trace(fig, cm1, positive_label, negative_label, 1, 1)
    add_confusion_trace(fig, cm2, positive_label, negative_label, 1, 2)

    add_labels_to_confusion_trace(fig, cm1, labels, plot_index=1)
    add_labels_to_confusion_trace(fig, cm2, labels, plot_index=2)

    add_lines_to_confusion(fig, 1)
    add_lines_to_confusion(fig, 2)

    add_ticks_to_confusion(fig, labels, 1)
    add_ticks_to_confusion(fig, labels, 2)

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def add_confusion_trace(
    fig: go.Figure,
    cm: NDArray[np.int_],
    positive_label: str,
    negative_label: str,
    row: int,
    col: int,
) -> None:
    """
    Add a confusion matrix trace to a plotly figure.

    Parameters
    ----------
    fig : go.Figure
        Figure to add a trace to.

    cm : np.ndarray[int]
        Confusion matrix data to add to the figure.

    positive_label : str
        Hover text for the positive class.

    negative_label : str
        Hover text for the negative class.

    row : int
        The row in fig to add the trace to.

    col : int
        The column in fig to add the trace to.

    """
    labels = ["0", "1"]
    hover_text = np.empty_like(cm, dtype=object)
    hover_text[0, 0] = (
        f"Truth: {negative_label}<br>Prediction: {positive_label}<br>Total: {cm[0, 0]}"
    )
    hover_text[0, 1] = (
        f"Truth: {negative_label}<br>Prediction: {negative_label}<br>Total: {cm[0, 1]}"
    )
    hover_text[1, 0] = (
        f"Truth: {positive_label}<br>Prediction: {positive_label}<br>Total: {cm[1, 0]}"
    )
    hover_text[1, 1] = (
        f"Truth: {positive_label}<br>Prediction: {negative_label}<br>Total: {cm[1, 1]}"
    )

    # Adding heatmaps to both subplots

    fig.add_trace(
        go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            showscale=False,
            text=hover_text,
            hoverinfo="text",
        ),
        row=row,
        col=col,
    )


def add_labels_to_confusion_trace(
    fig: go.Figure,
    cm: NDArray[np.int_],
    labels: list[str],
    plot_index: int,
) -> None:
    """
    Add permanent text labels to a confusion matrix trace.

    # Parameters
    ----------
    fig : go.Figure
        Figure to add labels to.

    cm : np.ndarray[int]
        Confusion matrix data to add to the figure.

    labels : list[str]
        Labels for the confusion matrix.

    plot_index : int
        The index of the plot to add the labels to.

    """
    for i, cm_row in enumerate(cm):
        for j, value in enumerate(cm_row):
            fig.add_annotation(
                {
                    "text": str(value),
                    "x": labels[j],
                    "y": labels[i],
                    "xref": f"x{plot_index}",
                    "yref": f"y{plot_index}",
                    "showarrow": False,
                    "font": {"color": get_font_color(value, cm)},
                }
            )


def add_lines_to_confusion(fig: go.Figure, plot_index: int) -> None:
    """
    Add aesthetic grid lines to a confusion matrix trace.

    Parameters
    ----------
    fig : go.Figure
        Figure to add lines to.

    plot_index : int
        Plot index to add lines to.

    """
    labels = ["0", "1"]
    fig.add_shape(
        type="rect",
        xref=f"x{plot_index}",
        yref=f"y{plot_index}",
        x0=-0.5,
        y0=-0.5,
        x1=len(labels) - 0.5,
        y1=len(labels) - 0.5,
        line={"color": "black", "width": 2},
    )
    fig.add_shape(
        type="line",
        xref=f"x{plot_index}",
        yref=f"y{plot_index}",
        x0=-0.5,
        y0=0.5,
        x1=len(labels) - 0.5,
        y1=0.5,
        line={"color": "black", "width": 1.5},
    )
    fig.add_shape(
        type="line",
        xref=f"x{plot_index}",
        yref=f"y{plot_index}",
        x0=0.5,
        y0=-0.5,
        x1=0.5,
        y1=1.5,
        line={"color": "black", "width": 1.5},
    )


def add_ticks_to_confusion(fig: go.Figure, labels: list[str], plot_index: int) -> None:
    """
    Add aesthetic ticks to a confusion matrix trace.

    Parameters
    ----------
    fig : go.Figure
        Figure to add ticks to.

    labels : list[str]
        Labels for the confusion matrix.

    plot_index : int
        Plot index to add ticks to.

    """
    fig.update_xaxes(
        title_text="Prediction",
        row=1,
        col=plot_index,
        tickmode="array",
        tickvals=[0, 1],
        ticktext=labels[::-1],
        ticks="outside",
        ticklen=5,  # Shorten tick length to touch grid lines
        tickwidth=1,
        tickcolor="black",
        mirror=True,  # Mirror ticks on both sides
    )
    fig.update_yaxes(
        title_text="Truth",
        row=1,
        col=plot_index,
        tickmode="array",
        tickvals=[0, 1],
        ticktext=labels,
        ticks="outside",
        ticklen=5,  # Shorten tick length to touch grid lines
        tickwidth=1,
        tickcolor="black",
        mirror=True,  # Mirror ticks on both sides
        title_standoff=0.3,
    )


def get_font_color(value: int, cm: NDArray[np.int_]) -> Literal["white", "dark blue"]:
    """
    Get the font color for a confusion matrix cell.

    This prevents white text from being displayed on a white background.

    Parameters
    ----------
    value : int
        Number of samples in the cell.

    cm : np.ndarray
        Confusion matrix data.

    Returns
    -------
    str
        The font color for the cell.


    """
    if value > cm.max() / 2:
        return "white"

    return "dark blue"


def binary_train_valid_test_confusion_plotly(
    y_train: ArrayLike,
    y_train_pred: ArrayLike,
    y_val: ArrayLike,
    y_val_pred: ArrayLike,
    y_test: ArrayLike,
    y_test_pred: ArrayLike,
    positive_label: str = "1",
    negative_label: str = "0",
    save_path: str | None = None,
) -> None:
    """
    Generate binary classification confusion matrix for train, validation and test data.

    These plots can be saved to an html file.
    The plots are interactive in the browser.
    Change the output method at the bottom of the function.

    Parameters
    ----------
    y_train : ArrayLike
        The binary labels for the training data. (bool or int)

    y_train_pred : ArrayLike
        Predicted binary labels for the training data. (bool or int)

    y_val : ArrayLike
        The binary labels for the validation data. (bool or int)

    y_val_pred : ArrayLike
        _Predicted binary labels for the validation data. (bool or int)

    y_test : ArrayLike
        The binary labels for the test data. (bool or int)

    y_test_pred : ArrayLike
        Predicted binary labels for the test data. (bool or int)

    positive_label : str, optional
        The label for the positive class.
        Alters the pos label displayed when hovering over confusion matrix with cursor.
        By default "1".

    negative_label : str, optional
        The label for the negative class.
        Alters the neg label displayed when hovering over confusion matrix with cursor.
        By default "0".

    save_path : str, optional
        Path to save the plotly figure as an html file.

    """
    labels = ["0", "1"]
    cm1 = confusion_matrix(y_train, y_train_pred, labels=[0, 1])
    cm1 = np.array([[cm1[0, 1], cm1[0, 0]], [cm1[1, 1], cm1[1, 0]]])

    cm2 = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
    cm2 = np.array([[cm2[0, 1], cm2[0, 0]], [cm2[1, 1], cm2[1, 0]]])

    cm3 = confusion_matrix(y_test, y_test_pred, labels=[0, 1])
    cm3 = np.array([[cm3[0, 1], cm3[0, 0]], [cm3[1, 1], cm3[1, 0]]])

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Training Matrix", "Validation Matrix", "Testing Matrix"],
        horizontal_spacing=0.15,
    )

    # Increase font size and height of titles
    for annotation in fig["layout"]["annotations"]:
        annotation["font"] = {"size": 18}
        annotation["y"] = 1.03

    fig.update_layout(
        autosize=False,
        width=1050,
        height=400,
    )

    add_confusion_trace(fig, cm1, positive_label, negative_label, 1, 1)
    add_confusion_trace(fig, cm2, positive_label, negative_label, 1, 2)
    add_confusion_trace(fig, cm3, positive_label, negative_label, 1, 3)

    add_labels_to_confusion_trace(fig, cm1, labels, plot_index=1)
    add_labels_to_confusion_trace(fig, cm2, labels, plot_index=2)
    add_labels_to_confusion_trace(fig, cm3, labels, plot_index=3)

    add_lines_to_confusion(fig, 1)
    add_lines_to_confusion(fig, 2)
    add_lines_to_confusion(fig, 3)

    add_ticks_to_confusion(fig, labels, 1)
    add_ticks_to_confusion(fig, labels, 2)
    add_ticks_to_confusion(fig, labels, 3)

    # Save as html file
    pio.write_html(
        fig,
        file="confusion_matrix.html",
        auto_open=False,
        include_plotlyjs="cdn",
    )

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
