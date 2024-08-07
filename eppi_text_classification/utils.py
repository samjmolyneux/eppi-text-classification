"""Utility functions for the eppi_text_classification package."""

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_and_names(
    word_features: list[str], min_df: int = 3, max_features: int = 75000
) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    """
    Get the tfidf scores and their corresponing feature names.

    This function assumes that the word_features are preprocessed.
    The TfidfVectorizer uses a non-whitespace token pattern, and as a result
    will do no processing or removal of punctuation or stop words.

    Parameters
    ----------
    word_features : list[str]
        List of preprocessed texts.

    min_df : int, optional
        Minimum document frequency. A given word feature must occur this many times
        throughout word_features to be included in the tfidf_scores. By default 3

    max_features : int, optional
        The maximum number of word features that vectorizer will create tfidf_scores
        for. See
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        for details.
        By default 75000

    Returns
    -------
    tuple[np.ndarray, list[str]]
        A tuple of tfidf_scores (samples, scores) and feature_names (samples,).

    """
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        token_pattern=r"(?u)\S+",  # Match any non-whitespace character
    )

    tfidf_scores = vectorizer.fit_transform(word_features)
    tfidf_scores = tfidf_scores.toarray()

    feature_names = vectorizer.get_feature_names_out()

    return tfidf_scores, feature_names
