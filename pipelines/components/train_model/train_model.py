
import argparse
import os

import json
import jsonpickle
import numpy as np
from lightgbm import LGBMClassifier
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

mname_to_mclass = {
    "SVC": SVC,
    "LGBMClassifier": LGBMClassifier,
    "RandomForestClassifier": RandomForestClassifier,
    "XGBClassifier": XGBClassifier,
}


def main():
    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--X_train",
        type=str,
        help="path to training data",
    )
    parser.add_argument(
        "--y_train",
        type=str,
        help="path to training labels",
    )
    parser.add_argument(
        "--model_parameters",
        type=str,
        help="path to model training parameters",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="path to trained model",
    )
    args = parser.parse_args()

    X_train = load_npz(os.path.join(args.X_train, "X_train.npz"))
    y_train = np.load(os.path.join(args.y_train, "y_train.npy"))
    model_params_path = os.path.join(args.model_parameters, "model_params.json")
    with open(model_params_path, "r") as file:
        json_model_parameters = json.load(file)
    model_parameters = jsonpickle.decode(json_model_parameters)

    model_class = mname_to_mclass[model_parameters.pop("model_name")]
    model = model_class(**model_parameters)

    model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(args.model, "model.joblib")) 

if __name__ == "__main__":
    main()
