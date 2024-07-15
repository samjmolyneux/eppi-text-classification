"""For performing run time validation of eppi_text_classification package."""

valid_models = ("LGBMClassifier", "RandomForestClassifier", "SVC", "XGBClassifier")


class InvalidModelError(Exception):
    """Exception for when an invalid model is passed."""

    def __init__(self, model: str) -> None:
        """Create an exception for an invalid model."""
        super().__init__(f"Model must be one of {valid_models}, but got {model}.")


def check_valid_model(model: str) -> None:
    """
    Check if the model is a valid model for the package.

    Parameters
    ----------
    model : str
        The model to check.

    Raises
    ------
    InvalidModelError

    """
    if model not in valid_models:
        raise InvalidModelError(model)
