from joblib import Parallel, delayed
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Considerations: Setting the joblib backend,
#                Choosing spacy model,


# TO DO:Loguru for processer count and chunksize
# TO DO Ability to change the process count

system_num_processes = os.cpu_count()


def lemmatize_pipe(doc):
    lemma_list = [
        token.lemma_.lower()
        for token in doc
        if (not token.is_stop) and (not token.is_punct)
    ]
    return lemma_list


def chunker(iterable, process_count):
    chunksize = -(-len(iterable) // process_count)  # ceiling division
    return (
        iterable[pos : pos + chunksize] for pos in range(0, len(iterable), chunksize)
    )


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(texts):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return [lemmatize_pipe(doc) for doc in nlp.pipe(texts, batch_size=25)]


def process_column(texts, process_count=system_num_processes):
    tasks = (
        delayed(process_chunk)(chunk)
        for chunk in chunker(texts, process_count=process_count)
    )
    result = Parallel(n_jobs=process_count, backend="loky")(tasks)
    return flatten(result)


def get_features(abstract_column, title_column):
    abstracts = process_column(abstract_column)
    titles = process_column(title_column)
    all_words = []
    for abstract, title in zip(abstracts, titles, strict=True):
        words = [f"t_{word}" for word in title] + [f"a_{word}" for word in abstract]
        all_words.append(words)

    return all_words


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


if __name__ == "__main__":
    df = pd.read_csv("data/raw/debunking_review.tsv", sep="\t")
    features, labels = get_features_and_labels(df)

    # df = pd.read_csv("../../data/raw/studytype_multiclass.tsv", sep="\t")
    # tfidf_scores, labels = get_features_and_labels(
    #     df, abstract_key="AB", title_key="TI"
    # )
