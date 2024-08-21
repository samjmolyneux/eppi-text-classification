"""For performing run time validation of eppi_text_classification package."""

import warnings

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


class InvalidDatabasePathError(Exception):
    """Exception for when an invalid database path is passed."""

    def __init__(self, path: str) -> None:
        """Create an exception for an invalid database path."""
        super().__init__(f"No database found at {path}")


def check_valid_database_url(database_url: str) -> None:
    """
    Check if there is a database for the given database_url.

    Parameters
    ----------
    database_url : str
        Url to the database.

    Raises
    ------
    ValueError
        Throws if the database path is bad.

    """
    database_path = database_url.split(":///", maxsplit=1)[1]
    if database_path.__contains__("site-packages"):
        msg = """It appears that you are trying to store the database in site-packages.
        This may lead to errors."""
        warnings.warn(msg, UserWarning, stacklevel=2)
    if database_path.__contains__("/shared/"):
        msg = """It appears that you may be trying to store the database in a shared
        directory. Using shared file systems may lead to SQLite Lock errors."""
        warnings.warn(msg, UserWarning, stacklevel=2)
