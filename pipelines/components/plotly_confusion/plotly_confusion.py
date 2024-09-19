
import argparse
import os
from pathlib import Path

from eppi_text_classification import (
    binary_train_valid_confusion_plotly,
    binary_train_valid_test_confusion_plotly,
)
from eppi_text_classification.utils import load_np_array_at_directory


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--y_train",
        type=str,
        help="path to y_train",
    )
    parser.add_argument(
        "--y_train_pred",
        type=str,
        help="path to y_train_pred",
    )
    parser.add_argument(
        "--y_val",
        type=str,
        help="path to y_val",
    )
    parser.add_argument(
        "--y_val_pred",
        type=str,
        help="path to y_val_pred",
    )
    parser.add_argument(
        "--y_test",
        type=str,
        help="path to y_test",
    )
    parser.add_argument(
        "--y_test_pred",
        type=str,
        help="path to y_test_pred",
    )
    parser.add_argument(
        "--confusion_plot",
        type=str,
        help="path to confusion plot",
    )
    args = parser.parse_args()
    y_train = load_np_array_at_directory(args.y_train)
    y_train_pred = load_np_array_at_directory(args.y_train_pred)
    y_val = load_np_array_at_directory(args.y_val)
    y_val_pred = load_np_array_at_directory(args.y_val_pred)
    y_test = load_np_array_at_directory(args.y_test)
    y_test_pred = load_np_array_at_directory(args.y_test_pred)

    save_path = Path(args.confusion_plot) / "confusion_plot.html"

    if not args.y_test:
        binary_train_valid_confusion_plotly(
            y_train,
            y_train_pred,
            y_val,
            y_val_pred,
            postive_label="Included",
            negative_label="Excluded",
            save_path=save_path,
        )
    else:
        binary_train_valid_test_confusion_plotly(
            y_train,
            y_train_pred,
            y_val,
            y_val_pred,
            y_test,
            y_test_pred,
            postive_label="Included",
            negative_label="Excluded",
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
