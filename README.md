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
│   └── utils.py
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
└── pyproject.toml
```

# Known Bugs

### 1. Inaccurate log decision plot
- Setting ShapPlotter.decision_plot() or ShapPlotter.single_decision_plot() with log_scale=True will result in an unaccurate decision plot.
### 2. Functions from opt.py not running
- Optimisation scripts still need hyperparmeter ranges tuned and whilst I'm making changes they may not run. 
