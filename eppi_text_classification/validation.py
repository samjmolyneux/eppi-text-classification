"""For performing run time validation of eppi_text_classification package."""

model_list = [
    "SVC",
    "LGBMClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "XGBClassifier",
]


class InvalidModelError(Exception):
    def __init__(self, model):
        super().__init__(f"Model must be one of {model_list}, but got {model}.")


def check_valid_model(model):
    if model.__class__.__name__ not in model_list:
        raise InvalidModelError(model)
