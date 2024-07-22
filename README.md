![Tests](https://github.com/samjmolyneux/eppi-text-classification/actions/workflows/Tests.yml/badge.svg)
![](https://byob.yarr.is/samjmolyneux/eppi-text-classification/ruff-badge)
![](https://byob.yarr.is/samjmolyneux/eppi-text-classification/mypy-badge)

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
<!-- directory-structure-start -->
```
.
├── README.md
├── data
│   └── raw
│       └── debunking_review.tsv
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
├── setup.py
├── tests
│   ├── check_install.py
│   └── test_00_smoke.py
└── tox.ini
```
<!-- directory-structure-end -->

# Known Bugs
- Documentation not up to date. 
