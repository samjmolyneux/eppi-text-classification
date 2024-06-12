from joblib import Parallel, delayed
import pandas as pd
import spacy
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import os


system_num_processes = os.cpu_count()

# nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def lemmatize_pipe(doc):
    lemma_list = [
        token.lemma_.lower()
        for token in doc
        if (not token.is_stop) and (not token.is_punct)
    ]
    return lemma_list


def chunker(iterable, total_length, process_count):
    chunksize = -(-total_length // process_count)  # ceiling division
    # it doesnt repeat the first elements if the final chunk doesnt fit
    return (
        iterable[pos : pos + chunksize] for pos in range(0, total_length, chunksize)
    )


def flatten(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(texts):
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    preproc_pipe = []

    # Batch_size needs managing for deployment
    [lemmatize_pipe(doc) for doc in nlp.pipe(texts, batch_size=100)]
    return preproc_pipe


def preprocess_parallel(texts, process_count=system_num_processes):
    executor = Parallel(
        n_jobs=process_count, backend="multiprocessing", prefer="processes"
    )
    do = delayed(process_chunk)
    tasks = (
        do(chunk) for chunk in chunker(texts, len(texts), process_count=process_count)
    )
    result = executor(tasks)
    return flatten(result)


def get_features(df):
    abstracts = preprocess_parallel(df["abstract"])
    titles = preprocess_parallel(df["title"])
    all_words = []
    for abstract, title in zip(abstracts, titles, strict=False):
        words = [f"t_{word}" for word in title] + [f"a_{word}" for word in abstract]
        all_words.append(words)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=75000,
        min_df=3,
        strip_accents="unicode",
        token_pattern=r"(?u)\S+",
    )
    features = vectorizer.fit_transform(all_words)
    return features


def get_features_and_labels(df, title_col_name="title", abstract_col_name="abstract"):
    df = df.rename(columns={title_col_name: "title", abstract_col_name: "abstract"})
    df = df.dropna(subset=["title", "abstract"], how="all")

    df["abstract"] = df["abstract"].astype(str)
    df["title"] = df["title"].astype(str)

    tfidf_scores = get_features(df)

    # Should make code here for loading labels of any type using a dictionary of some sort
    labels = df["included"].tolist()
    labels = [int(label) for label in labels]

    return tfidf_scores, labels


# NEED TO MAKE SURE THAT THIS CODE WONT DELETE THE EMPTY ELEMENTS FOR SOME REASON
# ALSO NEED TO UNDERSTAND WHATS GOING ON WITH THE FLATTEN
# ALSO NEED TO MAKE SURE THAT THEY COME BACK OUR IN THE CORRECT ORDER (THE MULTIPROCESSING MIGHT MESS UP THE ORDER OF THE ELEMENTS)
