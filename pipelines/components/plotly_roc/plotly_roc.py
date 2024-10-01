
import argparse
import os
from pathlib import Path

from eppi_text_classification import plotly_roc
from eppi_text_classification.utils import load_np_array_at_directory


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--y",
        type=str,
        help="path to labels",
    )
    parser.add_argument(
        "--y_pred_probs",
        type=str,
        help="path to the predicted probabilities",
    )
    parser.add_argument(
        "--roc_plot",
        type=str,
        help="path to the roc plot",
    )
    args = parser.parse_args()
    y = load_np_array_at_directory(args.y)
    y_pred_probs = load_np_array_at_directory(args.y_pred_probs)

    roc_plot_path = Path(args.roc_plot) / "roc_plot.html"
    plotly_roc(y, y_pred_probs, save_path=roc_plot_path)


if __name__ == "__main__":
    main()
