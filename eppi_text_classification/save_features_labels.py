"""Save word features and there associated labels."""

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import spacy
from joblib import Parallel, delayed

if TYPE_CHECKING:
    import pandas as pd


from multiprocessing import cpu_count

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
    process_count: int = system_num_processes,
) -> list[list[str]]:
    """
    Process a column of text data. E.g abstracts or titles.

    The function processes the data in parallel using the process_chunk function.
    Texts are lematized and stop words and punctuation are removed.

    Parameters
    ----------
    texts : Sequence[str] | pd.Series
        Any sequence like object containing strings.

    process_count : int, optional
        Number of processes to use when processing the data.
        By default equialent to system_num_processes.

    Returns
    -------
    list[list[str]]
        A list of strings for each data point in the column, lemmatized with punctuation
        and stop words removed.

    """
    tasks = (
        delayed(process_chunk)(chunk)
        for chunk in chunker(texts, process_count=process_count)
    )
    result = Parallel(n_jobs=process_count, backend="loky")(tasks)
    return flatten(result)


def get_features(
    abstract_column: Sequence[str],
    title_column: Sequence[str],
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

    Returns
    -------
    list[str]
        A list of processed word features for each data point.

    """
    abstracts = process_column(abstract_column)
    titles = process_column(title_column)
    features = []
    for abstract, title in zip(abstracts, titles, strict=True):
        words = [f"t_{word}" for word in title] + [f"a_{word}" for word in abstract]
        string = " ".join(words)
        features.append(string)

    return features


# TO DO: Get working for all data types
def get_labels(label_column: Sequence[int | str]) -> list[int]:
    """
    Turn all labels into integers.

    Parameters
    ----------
    label_column : Sequence
        Sequence of labels.

    Returns
    -------
    list[int]
        List of labels in integer format.

    """
    labels = [int(label) for label in label_column]
    return labels


def get_features_and_labels(
    df: "pd.DataFrame",
    title_key: str = "title",
    abstract_key: str = "abstract",
    y_key: str = "included",
) -> tuple[list[str], list[int]]:
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

    Returns
    -------
    tuple[list[str], list[int]]
        A tuple of word features followed by labels.

    """
    df = df.dropna(subset=[title_key, abstract_key], how="all")
    df[abstract_key] = df[abstract_key].astype(str)
    df[title_key] = df[title_key].astype(str)

    word_features = get_features(df[abstract_key].to_list(), df[title_key].to_list())

    labels = get_labels(df[y_key].to_list())

    return word_features, labels
