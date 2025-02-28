import argparse
import json
import os
from pathlib import Path

from scipy.sparse import save_npz

from eppi_text_classification import ShapPlotter
from eppi_text_classification.utils import (
    load_csr_at_directory,
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
        "--shap_values",
        type=str,
        help="path to shap_values",
    )
    parser.add_argument(
        "--shap_expected_value",
        type=str,
        help="path to shap expected_value",
    )
    args = parser.parse_args()
    X = load_csr_at_directory(args.X)
    feature_names = load_np_array_at_directory(args.feature_names, allow_pickle=True)
    model = load_joblib_model_at_directory(args.model)

    shap_plotter = ShapPlotter(
        model,
        X,
        feature_names,
    )

    shap_values_file = Path(args.shap_values) / "shap_values.npz"
    save_npz(shap_values_file, shap_plotter.shap_values)

    with open(
        os.path.join(args.shap_expected_value, "shap_expected_value.json"), "w"
    ) as file:
        json.dump(shap_plotter.expected_value, file)


if __name__ == "__main__":
    main()
