
import argparse
import os
import pickle
from pathlib import Path

from eppi_text_classification import ShapPlotter
from eppi_text_classification.utils import (
    load_joblib_model_at_directory,
    load_np_array_at_directory,
)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="path to model",
    )
    parser.add_argument(
        "--X",
        type=str,
        help="path to model",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="path to feature_names",
    )
    parser.add_argument(
        "--shap_plotter",
        type=str,
        help="path to shap_plotter",
    )
    args = parser.parse_args()
    X = load_np_array_at_directory(args.X)
    feature_names = load_np_array_at_directory(args.feature_names)
    model = load_joblib_model_at_directory(args.model)

    shap_plotter = ShapPlotter(
        model,
        X[:10],
        feature_names,
    )

    shap_plotter_file = Path(args.shap_plotter) / "shap_plotter.pkl"
    with shap_plotter_file.open("wb") as file:
        pickle.dump(shap_plotter, file)


if __name__ == "__main__":
    main()
