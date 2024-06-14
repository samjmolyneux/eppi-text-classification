from sklearn.feature_extraction.text import TfidfVectorizer


def get_tfidf_and_names(word_features, min_df=3, max_features=75000):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=max_features,
        min_df=min_df,
        strip_accents="unicode",
        token_pattern=r"(?u)\S+",
    )

    tfidf_scores = vectorizer.fit_transform(word_features)
    feature_names = vectorizer.get_feature_names_out()

    return tfidf_scores, feature_names
