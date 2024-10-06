![PyTest](https://github.com/samjmolyneux/eppi-text-classification/actions/workflows/pytest_tests.yml/badge.svg)
![Ruff](https://github.com/samjmolyneux/eppi-text-classification/actions/workflows/ruff_test.yml/badge.svg)
![MyPy](https://github.com/samjmolyneux/eppi-text-classification/actions/workflows/mypy_test.yml/badge.svg)

# Installation
Create a virtual environment.
```
conda create -n eppi_text python=3.11
conda activate eppi_text
```
Install.
```
pip3 install -e .
python3 -m spacy download en_core_web_sm
```

If you wish to run tests, you will need to install the test dependencies.
```
pip3 install -e ".[test]" 
```

# Setup
The workbench uses a database to track the hyperparameters and results of the hyperparameter search. To use this feature, you must have a database in an appropriate location.

### PC
On a local system, the OptunaHyperparameterOptimisation object will automatically find the optuna.db in this repo and use it. 

### Azure ML Studio/ Cloud service 
On Azure ML Studio, the database must be set more carefully to be on the same storage device as the compute instance. For example, when using notebooks, create a database at /mnt/tmp and set the db_url of OptunaHyperaparmeterOptimisation object appropriately:

```
cd /mnt/tmp
touch optuna.db
```
And set db_url appropriately in your script/notebook. In this case:
```
optimiser = OptunaHyperparameterOptimisation(
    db_url=f"sqlite:////mnt/tmp/optuna.db",
)
```

# Structure
<!-- directory-structure-start -->
```
.
├── README.md
├── data
│   └── raw
│       └── debunking_review.tsv
├── eppi_text_classification
│   ├── __init__.py
│   ├── hyperparameter_schemas.py
│   ├── opt.py
│   ├── plotly_confusion.py
│   ├── plotly_roc.py
│   ├── plots.py
│   ├── predict.py
│   ├── save_features_labels.py
│   ├── shap_colors
│   │   ├── __init__.py
│   │   ├── _colorconv.py
│   │   └── _colors.py
│   ├── shap_plotter.py
│   ├── utils.py
│   └── validation.py
├── notebooks
│   ├── lgbm
│   │   └── lgbm_binary.ipynb
│   ├── random_forest
│   │   └── random_forest_binary.ipynb
│   ├── svm
│   │   └── svm_binary.ipynb
│   └── xgboost
│       └── xgboost_binary.ipynb
├── optuna.db
├── pipelines
│   ├── change_file_names.py
│   ├── components
│   │   ├── calculate_shap_values
│   │   │   └── calculate_shap_values.py
│   │   ├── create_bar_plot
│   │   │   └── create_bar_plot.py
│   │   ├── create_decision_plot
│   │   │   └── create_decision_plot.py
│   │   ├── create_dot_plot
│   │   │   └── create_dot_plot.py
│   │   ├── create_shapplotter
│   │   │   └── create_shapplotter.py
│   │   ├── dot_plot
│   │   │   └── dot_plot.py
│   │   ├── get_labels
│   │   │   └── get_labels.py
│   │   ├── get_labels_file
│   │   │   └── get_labels.py
│   │   ├── get_tfidf_and_feature_names
│   │   │   └── get_tfidf_and_feature_names.py
│   │   ├── get_threshold
│   │   │   └── get_threshold.py
│   │   ├── hyperparameter_search
│   │   │   └── optuna_search.py
│   │   ├── plotly_confusion
│   │   │   └── plotly_confusion.py
│   │   ├── plotly_roc
│   │   │   └── plotly_roc.py
│   │   ├── predict_scores
│   │   │   └── predict_scores.py
│   │   ├── process_data
│   │   │   └── data_prep.py
│   │   ├── splice_data
│   │   │   └── splice_data.py
│   │   ├── split_data
│   │   │   └── split_data.py
│   │   ├── split_with_primitive
│   │   │   └── split_with_primitive.py
│   │   ├── threshold_predict
│   │   │   └── threshold_predict.py
│   │   ├── train_model
│   │   │   └── train_model.py
│   │   └── view_html_image
│   │       └── view_html_image.py
│   ├── create_first_pipeline.ipynb
│   ├── dependencies
│   │   ├── conda.yaml
│   │   └── display_image_env.yaml
│   ├── load_azure_ml.py
│   ├── push_bar_plot_to_azure.py
│   ├── push_create_shap_plotter_to_azure.py
│   ├── push_debunking_to_az_to_azure.py
│   ├── push_decision_plot_to_azure.py
│   ├── push_dot_plot_to_azure.py
│   ├── push_get_labels_to_azure.py
│   ├── push_get_raw_threshold_to_azure.py
│   ├── push_get_tfidf_and_feature_names_to_azure.py
│   ├── push_hyperparam_search_to_azure.py
│   ├── push_mlflow_env_to_az_to_azure.py
│   ├── push_package_env_to_az_to_azure.py
│   ├── push_plotly_confusion_to_azure.py
│   ├── push_plotly_roc_to_azure.py
│   ├── push_predict_scores_to_azure.py
│   ├── push_splice_data_component_to_azure.py
│   ├── push_split_data_to_azure.py
│   ├── push_threshold_predict_to_azure.py
│   ├── push_train_model_to_azure.py
│   ├── push_view_html_image_to_azure.py
│   └── user_inputs
│       ├── abstract_column_name.json
│       ├── false.json
│       ├── float_1.json
│       ├── float_10.json
│       ├── hyperparam_search_input.json
│       ├── label_column.json
│       ├── label_column_name.json
│       ├── positive_class_value.json
│       ├── test_size_025.json
│       ├── test_size_05.json
│       └── title_column_name.json
├── pyproject.toml
├── setup.py
├── tests
│   ├── check_install.py
│   └── test_00_smoke.py
└── tox.ini
```
<!-- directory-structure-end -->

# Known Bugs
- None
