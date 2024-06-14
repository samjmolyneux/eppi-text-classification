from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import optuna


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


def delete_optuna_study(study_name):
    root_path = Path(Path(__file__).resolve()).parent.parent
    db_storage_url = f"sqlite:///{root_path}/optuna.db"

    optuna.delete_study(study_name=study_name, storage=db_storage_url)
