"""Save word features and there associated labels."""

from collections.abc import Iterator, Sequence
from multiprocessing import cpu_count
from typing import TYPE_CHECKING, Any

import numpy as np
import spacy
from joblib import Parallel, delayed
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    import pandas as pd
    from scipy.sparse import csr_matrix

# Considerations: Setting the joblib backend,
#                Choosing spacy model,


# TO DO:Loguru for processer count and chunksize
# TO DO Ability to change the process count

system_num_processes = cpu_count()


def lemmatize_pipe(doc: spacy.tokens.Doc) -> list[str]:
    """
    Lemmatize a spacy doc and remove stop words and punctuation.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        A converted doc for an individual data point.

    Returns
    -------
    list[str]
        A list of lemmatized words.

    """
    lemma_list = [
        token.lemma_.lower()
        for token in doc
        if (not token.is_stop) and (not token.is_punct)
    ]
    return lemma_list


def chunker(object_list: Sequence[Any], process_count: int) -> Iterator[Sequence[Any]]:
    """
    Split a sequence into equal chunks for processing by multiple processes.

    Parameters
    ----------
    object_list : Sequence[int]
        Any sequence like object containing data to be processed.

    process_count : int
        The number of available processes.

    Returns
    -------
    Iterator[Sequence]
        Iterator of chunks of the object_list.

    """
    chunksize = -(-len(object_list) // process_count)  # ceiling division
    return (
        object_list[pos : pos + chunksize]
        for pos in range(0, len(object_list), chunksize)
    )


def flatten(list_of_lists: list[list[Any]]) -> list[Any]:
    """
    Take a list of lists and join into a single list.

    Parameters
    ----------
    list_of_lists : list[list]
        List of lists.

    Returns
    -------
    list
        List with all lowest level lists joined.

    """
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(texts: Sequence[str]) -> list[list[str]]:
    """
    Lemmatize and process a list of texts.

    This function is designed to be used in parallel processing.

    Parameters
    ----------
    texts : Seqeunce[str]
        A Sequence of texts to be processed.

    Returns
    -------
    list[list[str]]
        A list of lemmatized words for each text.

    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return [lemmatize_pipe(doc) for doc in nlp.pipe(texts, batch_size=25)]


def process_column(
    texts: Sequence[str],
    num_processes: int = system_num_processes,
) -> list[list[str]]:
    """
    Process a column of text data. E.g abstracts or titles.

    The function processes the data in parallel using the process_chunk function.
    Texts are lematized and stop words and punctuation are removed.

    Parameters
    ----------
    texts : Sequence[str] | pd.Series
        Any sequence like object containing strings.

    num_processes : int, optional
        Number of processes to use when processing the data.
        By default is system_num_processes, which uses all available processes.

    Returns
    -------
    list[list[str]]
        A list of strings for each data point in the column, lemmatized with punctuation
        and stop words removed.

    """
    # Joblib has a bug when num_processes is -1, we catch this case here
    if num_processes == -1:
        num_processes = system_num_processes

    print(f"number of processes: {num_processes}")
    tasks = (
        delayed(process_chunk)(chunk)
        for chunk in chunker(texts, process_count=num_processes)
    )
    result = Parallel(n_jobs=num_processes, backend="loky")(tasks)
    return flatten(result)


def get_features(
    abstract_column: Sequence[str],
    title_column: Sequence[str],
    num_processes: int = system_num_processes,
) -> list[str]:
    """
    Get the word features for a given abstract and title column.

    The function processes the abstract and title columns in parallel,
    removing stop words and punctuation and lemmatizing the text.
    The abstract and title words are then combined into a single string
    to get the word features. To distinguish between title and abstract words,
    a_ is added to the start of each abstract word and t_ to the start of
    each title word.

    Parameters
    ----------
    abstract_column : Sequence[str] | pd.Series
        A sequence of strings containing abstracts.

    title_column : Sequence[str]
        A sequence of strings containing titles.

    num_processes : int, optional
        Number of processes to use when processing the data.
        By default is system_num_processes, which uses all available processes.

    Returns
    -------
    list[str]
        A list of processed word features for each data point.

    """
    abstracts = process_column(abstract_column, num_processes=num_processes)
    titles = process_column(title_column, num_processes=num_processes)
    features = []
    for abstract, title in zip(abstracts, titles, strict=True):
        words = [f"t_{word}" for word in title] + [f"a_{word}" for word in abstract]
        string = " ".join(words)
        features.append(string)

    return features


# TO DO: Get working for all data types
def get_labels(label_column: Sequence[int | str]) -> NDArray[np.int64]:
    """
    Turn all labels into integers.

    Parameters
    ----------
    label_column : Sequence
        Sequence of labels.

    Returns
    -------
    NDArray[np.int64]:
        Array of labels in integer format.

    """
    labels = np.array([int(label) for label in label_column], dtype=int)
    return labels


def get_features_and_labels(
    df: "pd.DataFrame",
    title_key: str = "title",
    abstract_key: str = "abstract",
    y_key: str = "included",
    num_processes: int = system_num_processes,
) -> tuple[list[str], NDArray[np.int64]]:
    """
    Get the title and abstract word features and labels from a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the columns.

    title_key : str, optional
        Dataframe column key for the title column, by default "title".

    abstract_key : str, optional
        Dataframe column key for the abstract column, by default "abstract".

    y_key : str, optional
        Dataframe column key for the label column, by default "included"

    num_processes : int, optional
        Number of processes to use when processing the data.
        By default is system_num_processes, which uses all available processes.

    Returns
    -------
    tuple[list[str],  NDArray[np.int64]]
        A tuple of word features followed by labels.

    """
    df = df.dropna(subset=[title_key, abstract_key], how="all")
    df[abstract_key] = df[abstract_key].astype(str)
    df[title_key] = df[title_key].astype(str)

    word_features = get_features(
        abstract_column=df[abstract_key].to_list(),
        title_column=df[title_key].to_list(),
        num_processes=num_processes,
    )

    labels = get_labels(df[y_key].to_list())

    return word_features, labels


def get_tfidf_and_names(
    word_features: list[str], min_df: int = 3, max_features: int = 75000
) -> tuple["csr_matrix", NDArray[np.str_]]:
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
    tuple[np.ndarray[float], list[str]]
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

    feature_names = vectorizer.get_feature_names_out()

    return tfidf_scores, feature_names
