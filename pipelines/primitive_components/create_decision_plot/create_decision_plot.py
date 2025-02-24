
import argparse
import json
import os
from pathlib import Path

from eppi_text_classification.shap_plotter import DecisionPlot
from eppi_text_classification.utils import (
    load_csr_at_directory,
    load_np_array_at_directory,
    load_value_from_json_at_directory,
)


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expected_shap_value",
        type=str,
        help="path to expected shap value",
    )
    parser.add_argument(
        "--threshold",
        type=str,
        help="path to decision threshold for your model",
    )
    parser.add_argument(
        "--shap_values",
        type=str,
        help="path to shap_plotter",
    )
    parser.add_argument(
        "--X",
        type=str,
        help="path to X data to explain model on",
    )
    parser.add_argument(
        "--feature_names",
        type=str,
        help="path to features names",
    )
    parser.add_argument(
        "--num_display",
        type=int,
        help="number of features to display on plot",
        default=10,
    )
    parser.add_argument(
        "--log_scale",
        type=bool,
        help="bool of whether to display plot along log scale",
        default=False,
    )
    parser.add_argument(
        "--decision_plot",
        type=str,
        help="path to decision plot",
    )
    args = parser.parse_args()

    expected_shap_value = float(
        load_value_from_json_at_directory(args.expected_shap_value)
    )
    threshold = float(load_value_from_json_at_directory(args.threshold))
    shap_values = load_csr_at_directory(args.shap_values)
    X = load_csr_at_directory(args.X)
    feature_names = load_np_array_at_directory(args.feature_names, allow_pickle=True)
    num_display = args.num_display
    log_scale = args.log_scale

    decision_plot = DecisionPlot(
        expected_value=expected_shap_value,
        threshold=threshold,
        shap_values=shap_values,
        X_test=X,
        feature_names=feature_names,
        num_display=num_display,
        log_scale=log_scale,
    )

    print(f"expected_shap_value : {expected_shap_value}")
    print(f"threshold : {threshold}")
    print(f"feature_names : {feature_names.shape}")
    print(f"shap_values : {shap_values.shape}")
    print(f"X : {X.shape}")
    decision_plot_path = Path(args.decision_plot) / "decision_plot.png"
    decision_plot.save(decision_plot_path)


if __name__ == "__main__":
    main()
