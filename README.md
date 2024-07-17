# Installation
Create a virtual environment.
```
conda create -n eppi_text python=3.11
conda activate eppi_text
```
Install.
```
pip3 install -e . --config-settings editable_mode=strict 
python3 -m spacy download en_core_web_sm
```

# Structure
```
├── README.md
├── data
│   ├── processed
│   └── raw
├── eppi_text_classification
│   ├── __init__.py
│   ├── opt.py
│   ├── plotly_confusion.py
│   ├── plotly_roc.py
│   ├── plots.py
│   ├── predict.py
│   ├── save_features_labels.py
│   ├── shap_colors
│   │   ├── __init__.py
│   │   ├── _colorconv.py
│   │   └── _colors.py
│   ├── shap_plotter.py
│   ├── utils.py
│   └── validation.py
├── notebooks
│   ├── lgbm
│   │   └── lgbm_binary.ipynb
│   ├── random_forest
│   │   └── random_forest_binary.ipynb
│   ├── svm
│   │   └── svm_binary.ipynb
│   └── xgboost
│       └── xgboost_binary.ipynb
├── optuna.db
├── pyproject.toml
└── tests
    └── test_00_smoke.py
```
[//]: # (This is the method I used : tree -I "*.ris|*.tsv|*.html|eppi_text_classification.egg-info|__pycache__|Cochrane heart reviews|build|legacy_funcs|prototyping*")

# Known Bugs
There are no known bugs.

<!-- directory-structure-start -->
# Directory Structure
```
.
├── DIRECTORY_STRUCTURE.md
├── README.md
├── eppi_text_classification
│   ├── __init__.py
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
├── pyproject.toml
└── tests
    └── test_00_smoke.py
```
<!-- directory-structure-end -->
