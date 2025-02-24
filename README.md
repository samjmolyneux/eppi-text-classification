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

# Known Bugs
- None
