import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from eppi_text_classification.predict import predict_scores
from eppi_text_classification.train import train


def _learning_curve_for_one_proportion(
    proportion,
    tfidf_scores,
    labels,
    model_name,
    model_params,
    kf_splits,  # list( (train_idx, val_idx) ), pre-computed once
):
    train_auc_scores = []
    val_auc_scores = []
    fold_train_sizes = []

    for train_idx, val_idx in kf_splits:
        # Take a proportion of the training folds
        # Train test split uses ceil of the test proportion, so need 13 of each class
        # To guarantee that we have one of each class in the 0.1 proportion
        # We are using the test size to determine the size of the training set
        # As a result we take the test from train_test_split and use it for training
        X_train = tfidf_scores[train_idx]
        y_train = labels[train_idx]
        if proportion < 1.0:
            _, X_train, _, y_train = train_test_split(
                X_train,
                y_train,
                test_size=proportion,
                random_state=42,
                stratify=y_train,
            )

        X_val = tfidf_scores[val_idx]
        y_val = labels[val_idx]

        fold_train_sizes.append(len(y_train))

        clf = train(model_name, model_params, X_train, y_train)
        train_auc = roc_auc_score(y_train, predict_scores(clf, X_train))
        val_auc = roc_auc_score(y_val, predict_scores(clf, X_val))

        train_auc_scores.append(train_auc)
        val_auc_scores.append(val_auc)

    mean_train_size = np.mean(fold_train_sizes)
    return mean_train_size, train_auc_scores, val_auc_scores


# TODO: test that the test result is calculated by testing on the full dataset, not just a fold of the smaller proportion.
def get_learning_curve_data(
    tfidf_scores,
    labels,
    model_name,
    model_params,
    nfolds: int = 5,
    proportions=None,
    n_jobs: int = -1,  # -1 = use all CPU cores
):
    if proportions is None:
        proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Pre-compute CV splits once so every worker sees the identical folds
    kf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=42)
    kf_splits = list(kf.split(tfidf_scores, labels))

    # Launch one job per proportion
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_learning_curve_for_one_proportion)(
            prop,
            tfidf_scores,
            labels,
            model_name,
            model_params,
            kf_splits,
        )
        for prop in proportions
    )

    # Unpack results preserving the order of `proportions`
    train_sizes, train_curve_data, val_curve_data = map(
        list, zip(*results, strict=True)
    )
    return train_sizes, train_curve_data, val_curve_data


# def get_learning_curve_data(
#     tfidf_scores,
#     labels,
#     model_name,
#     model_params,
#     nfolds=5,
#     proportions=None,
# ):
#     if proportions is None:
#         proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#     train_sizes = []

#     # We use nfolds 0s to make the curve data homogenous, as all following
#     # lists will have an auc score for each fold.
#     train_curve_data = []
#     val_curve_data = []

#     kf = StratifiedKFold(n_splits=nfolds, shuffle=True)
#     for proportion in proportions:
#         train_auc_scores = []
#         val_auc_scores = []
#         fold_train_sizes = []

#         for _, (train_idx, val_idx) in enumerate(kf.split(tfidf_scores, labels)):
#             train_idx_slice = train_idx[: int(len(train_idx) * proportion)]

#             fold_train_sizes.append(len(train_idx_slice))

#             X_train = tfidf_scores[train_idx_slice]
#             X_val = tfidf_scores[val_idx]

#             y_train = labels[train_idx_slice]
#             y_val = labels[val_idx]

#             clf = train(model_name, model_params, X_train, y_train)

#             y_train_scores = predict_scores(clf, X_train)
#             y_val_scores = predict_scores(clf, X_val)

#             train_auc = roc_auc_score(y_train, y_train_scores)
#             train_auc_scores.append(train_auc)

#             val_auc = roc_auc_score(y_val, y_val_scores)
#             val_auc_scores.append(val_auc)

#         train_sizes.append(np.mean(fold_train_sizes))
#         train_curve_data.append(train_auc_scores)
#         val_curve_data.append(val_auc_scores)

#     return train_sizes, train_curve_data, val_curve_data


def learning_curve(
    tfidf_scores,
    labels,
    model_name,
    model_params,
    savepath,
    nfolds=5,
    proportions=None,
):
    if proportions is None:
        proportions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    train_sizes, train_curve_data, val_curve_data = get_learning_curve_data(
        tfidf_scores=tfidf_scores,
        labels=labels,
        model_name=model_name,
        model_params=model_params,
        nfolds=nfolds,
        proportions=proportions,
    )

    train_curve_ys = np.mean(train_curve_data, axis=1).tolist()
    val_curve_ys = np.mean(val_curve_data, axis=1).tolist()

    train_lower_bound_ys = np.min(train_curve_data, axis=1).tolist()
    val_lower_bound_ys = np.min(val_curve_data, axis=1).tolist()

    train_upper_bound_ys = np.max(train_curve_data, axis=1).tolist()
    val_upper_bound_ys = np.max(val_curve_data, axis=1).tolist()

    plot_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Learning Curve with Shaded AUC Bounds</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>

                body {{
                    margin: 0;
                    padding: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}

                #plot-container {{
                    width: 80%;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin: 1rem;
                }}

                #plot {{
                    width: 100%;
                    aspect-ratio: 16 / 9;
                }}


            </style>
        </head>
        <body>
        <div id="plot-container">
            <div id="plot"></div>
        </div>
        <script>

            var trainSizes = {np.array(train_sizes).tolist()};

            var trainCurveYs = {train_curve_ys};
            var valCurveYs = {val_curve_ys};

            var trainCurveLowerBoundYs = {train_lower_bound_ys};
            var valCurveLowerBoundYs = {val_lower_bound_ys};

            var trainCurveUpperBoundYs = {train_upper_bound_ys};
            var valCurveUpperBoundYs = {val_upper_bound_ys};

            var valCurveData = {np.array(val_curve_data).tolist()};
            console.log(valCurveData);

            var data = [];

            // --- Training Curve Shaded Area ---
            // Upper bound for training (invisible trace used for fill reference)
            //data.push({{
            //x: trainSizes,
            //y: trainCurveUpperBoundYs,
            //mode: "lines",
            //line: {{ width: 0 }},
            //showlegend: false,
            //hoverinfo: "skip"
            //}});
//
//            // Lower bound for training, fill to the previous trace (upper bound)
//            data.push({{
//            x: trainSizes,
//            y: trainCurveLowerBoundYs,
//            mode: "lines",
//            line: {{ width: 0 }},
//            fill: "tonexty",
//            fillcolor: "rgba(0, 0, 255, 0.2)", // semi-transparent blue
//            showlegend: false,
//            hoverinfo: "skip"
//            }});
//
//            // Average training curve
//            data.push({{
//            x: trainSizes,
//            y: trainCurveYs,
//            mode: "lines+markers",
//            name: "Training AUC",
//            line: {{ color: "blue" }},
//            marker: {{ size: 8 }}
//            }});

            // --- Validation Curve Shaded Area ---
            // Upper bound for validation
            data.push({{
                x: trainSizes,
                y: valCurveUpperBoundYs,
                mode: "lines",
                line: {{ width: 0 }},
                showlegend: false,
                hoverinfo: "skip"
            }});

            // Lower bound for validation, fill to previous trace (upper bound)
            data.push({{
                x: trainSizes,
                y: valCurveLowerBoundYs,
                mode: "lines",
                line: {{ width: 0 }},
                fill: "tonexty",
                fillcolor: "rgba(255, 0, 0, 0.2)", // semi-transparent red
                showlegend: false,
                hoverinfo: "skip"
            }});

            // Average validation curve
            data.push({{
                x: trainSizes,
                y: valCurveYs,
                mode: "lines+markers",
                name: "Validation AUC",
                line: {{ color: "red" }},
                marker: {{ size: 8 }}
            }});

            console.log(trainSizes);

            for (let i = 0; i < trainSizes.length; i++) {{
                data.push({{
                    type: "scatter",
                    mode: "markers",
                    x: new Array(valCurveData[i].length).fill(trainSizes[i]),
                    y: valCurveData[i],
                    marker: {{ color: "rgba(255, 0, 0, 0.3)" }},
                    hoverinfo: "skip",
                    showlegend: false,
                }});

            }}

            // --- Layout ---
            var layout = {{
            title: {{
                text: "Learning Curve with Shaded AUC Bounds",
                xanchor: "center",
                x: 0.45,
                xref: "x domain",
            }},
            xaxis: {{
                title: {{
                    text: "Training Set Size",
                }},
            }},
            yaxis: {{
                title: {{
                    text: "ROC AUC",
                    standoff: 10,
                }},
                automargin: true,
            }},
            template: "plotly_white",
            dragmode: "pan",
            }};

            var config = {{
                responsive: true,
                scrollZoom: false,
                showLink: true,
                plotlyServerURL: "https://chart-studio.plotly.com",
                modeBarButtons: [["toImage", "zoom2d", "pan2d", "autoScale2d"]],
                displaylogo: false,
                displayModeBar: "always",
                toImageButtonOptions: {{
                format: "png",
                filename: "eppi-learning-curve",
                height: 720,
                width: 1480,
                scale: 3
                }}
            }};

            Plotly.newPlot("plot", data, layout, config=config);
        </script>
        </body>
        </html>
    """

    with open(savepath, "w") as f:
        f.write(plot_html)

    print("Saved learning curve plot to", savepath)
