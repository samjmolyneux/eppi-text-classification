"""Save word features and there associated labels."""

import os
from collections.abc import Iterator

import spacy
from joblib import Parallel, delayed

# Considerations: Setting the joblib backend,
#                Choosing spacy model,


# TO DO:Loguru for processer count and chunksize
# TO DO Ability to change the process count

system_num_processes = os.cpu_count()


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


def chunker(object_list: list, process_count: int) -> Iterator[list]:
    """
    Split a sequence into equal chunks for processing by multiple processes.

    Parameters
    ----------
    object_list : list
        Any sequence like object containing data to be processed.

    process_count : int
        The number of available processes.

    Returns
    -------
    Iterator[list]
        Iterator of chunks of the object_list.

    """
    chunksize = -(-len(object_list) // process_count)  # ceiling division
    return (
        object_list[pos : pos + chunksize]
        for pos in range(0, len(object_list), chunksize)
    )


def flatten(list_of_lists: list[list]) -> list:
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


def process_chunk(texts: list[str]) -> list[list[str]]:
    """
    Lemmatize and process a list of texts.

    This function is designed to be used in parallel processing.

    Parameters
    ----------
    texts : list[str]
        A list of texts to be processed.

    Returns
    -------
    list[list[str]]
        A list of lemmatized words for each text.

    """
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return [lemmatize_pipe(doc) for doc in nlp.pipe(texts, batch_size=25)]


def process_column(texts: list[str], process_count: int = system_num_processes):
    tasks = (
        delayed(process_chunk)(chunk)
        for chunk in chunker(texts, process_count=process_count)
    )
    result = Parallel(n_jobs=process_count, backend="loky")(tasks)
    return flatten(result)


def get_features(abstract_column, title_column):
    abstracts = process_column(abstract_column)
    titles = process_column(title_column)
    features = []
    for abstract, title in zip(abstracts, titles, strict=True):
        words = [f"t_{word}" for word in title] + [f"a_{word}" for word in abstract]
        string = " ".join(words)
        features.append(string)

    return features


# TO DO: Get working for all data types
def get_labels(label_column):
    labels = label_column.tolist()
    labels = [int(label) for label in labels]
    return labels


def get_features_and_labels(
    df, title_key="title", abstract_key="abstract", y_key="included"
):
    df = df.dropna(subset=[title_key, abstract_key], how="all")
    df[abstract_key] = df[abstract_key].astype(str)
    df[title_key] = df[title_key].astype(str)

    word_features = get_features(df[abstract_key], df[title_key])

    labels = get_labels(df[y_key])

    return word_features, labels
