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

# Find Single Model

## Inputs

### labelled_data_path

> **Type:** String

> **Description** The path to the labelled data tsv, relative to the selected working container url of the production blob storage as set by the `working_container_url` input. This labelled data is used for training the model, finding the best model via n-fold cross-validation and generating statistics on the performance of the trained model.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

### unlabelled_data_path

> **Type:** String

> **Description:** The path to the unlabelled data tsv, relative to the selected working container url of the production blob storage as set by the `working_container_url` input. This unlabelled data will be classified by the trained model.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

### title_header

> **Type:** String

> **Description:** The header of the title column in the tsv file.

> **Implementation**: SET_FROM_DATA_FACTORY

### abstract_header

> **Type:** String

> **Description:** The header of the abstract column in the tsv file.

> **Implementation**: SET_FROM_DATA_FACTORY

### label_header

> **Type:** String

> **Description:** The header of label column in the tsv file.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

### positive_class_value

> **Type:** String

> **Description:** The value in column headed by `label_header` that the model should consider a positive.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

(Need to give some more details on why I think that the two above may require internal logic)

### model_name

> **Type:** String

> **Description:** The name of the model that the user would like to train for classification.

> **Choices:** lightgbm | RandomForestClassifier | xgboost | SVC

> **Implementation**: EXPOSE_IMMEDIATELY

> **notes:** a different type of xgboost can be selected by carefully selecting hyperparameter ranges. It is very good so I will probably expost it as it's own name when I get around to it. It will be called "xgboostLinear"

### hparam_search_ranges_path

> **Type:** String

> **Description:** A path to json with the hyperparameter search ranges for selecting the model. This follows a particular format that I will document later.

> **Implementation**: ADVANCED

### max_n_search_iterations

> **Type:** Integer

> **Description:** The number of iterations of hyperaparameter search the user would like to do. The best model will be selected out of all iterations.

> **Implementation**: ADVANCED

### nfolds

> **Type:** Integer

> **Description:** The number of folds to use in cross-validation.

> **Implementation**: ADVANCED

### num_cv_repeats

> **Type:** Integer

> **Description:** When n-fold cross-valdation is done with a small number of samples, the result can vary massively based on how the folds are selected. This can result in suboptimal models being selected due to the instability of the method. To combat this, we can repeat cross-validation with many different seeds to reduce the statistical variation.

> **Implementation**: ADVANCED

### timeout

> **Type:** Integer

> **Description:** The time in seconds after which the search will be terminated if it is still running.

> **Implementation**: ADVANCED

### use_early_terminator

> **Type:** Boolean

> **Description:** A gaussian mixture model that measures the statistical variation of results and measure the likelihood that a better model exists. When that likelihood is sufficiently low, it terminates the search.

> **Implementation**: ADVANCED

### max_stagnation_iterations

> **Type:** Integer

> **Description:** If a the search does not find a new best model after this many iterations since the last best model, then the search terminates.

> **Implementation**: ADVANCED

### wilcoxon_trial_pruner_threshold

> **Type:** Number

> **Description:** When set, a wilcoxon trial is performed on the results of the hyperparameter search vs the current best. If the new searches first few results are significantly worse, it is pruned.

> **Implementation**: ADVANCED

### use_worse_than_first_two_pruner

> **Type:** Boolean

> **Description:** When set to true, if the first two results a cross-validation iteration of the hyperparameter search are worse than the best result, then the trial is pruned. Results in atleast 33% time reduction of search.

> **Implementation**: ADVANCED

### shap_num_display

> **Type:** Integer

> **Description:** The top `shap_num_display` features will be used in the in the shap model explainability plots.

> **Implementation**: ADVANCED

### working_container_url

> **Type:** String

> **Description:** The url of the working container in the production blob storage. All paths, such as `labelled_data_path`, `unlabelled_data_path` and `output_container_path` are relative to this. The pipeline will not have access anything in the blob storage that is not nested within this container.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

### output_container_path

> **Type:** String

> **Description:** The path to save data and results of the pipeline. This should be the relative path to the working_container_url.

> **Implementation**: MAY_REQUIRE_INTERNAL_LOGIC

### managed_identity_client_id

> **Type:** String

> **Description:** The id of the managed identity. This managed identity provides the cluster with access to the production blob storage.

> **Implementation**: SET_FROM_DATA_FACTORY
