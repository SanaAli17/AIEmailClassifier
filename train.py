"""
train.py

Trains all six models used by the app.

Reads database/mail_dataset.csv with columns: text, label, category.
  Stage 1 (spam vs ham): uses every row.
  Stage 2 (categories):  uses rows where `category` is filled in.

All trained models are saved as .pkl files in trained_models/.

Run:
    python train.py
    
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from models.preprocessor import preprocess, download_nltk_data


# Paths and constants                                                         

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
MODELS_DIR   = os.path.join(BASE_DIR, "trained_models")

MAIL_DATASET_PATH = os.path.join(DATABASE_DIR, "mail_dataset.csv")

CATEGORY_LABELS = ("work", "personal", "promotion")

RANDOM_STATE = 42
TEST_SIZE    = 0.20

# Training                                                                    

def build_classifiers():
    """Return a fresh dict of the three model types."""
    return {
        "Naive Bayes": MultinomialNB(),
        "KNN":         KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "MLP":         MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(name, y_true, y_pred, pos_label="spam"):
    """Print and return accuracy / precision / recall / F1."""
    if pos_label is None:
        # Multi-class (categories): use the macro average.
        kwargs = {"average": "macro", "zero_division": 0}
    else:
        # Binary (spam vs ham): treat `spam` as the positive class.
        kwargs = {"pos_label": pos_label, "zero_division": 0}

    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall":    recall_score(y_true,    y_pred, **kwargs),
        "f1":        f1_score(y_true,        y_pred, **kwargs),
    }
    print(
        f"  {name:<14}  "
        f"acc={metrics['accuracy']:.4f}  "
        f"prec={metrics['precision']:.4f}  "
        f"rec={metrics['recall']:.4f}  "
        f"f1={metrics['f1']:.4f}"
    )
    return metrics


def save_pickle(filename, obj):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(obj, path)
    print(f"[save] {path}")


def train_spam_ham(df):

    """Stage 1: spam vs ham. Uses every row."""

    df = df[df["label"].isin(["ham", "spam"])].copy()
    df["clean"] = df["text"].astype(str).apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    print(
        f"[data] stage 1 rows: {len(df):,} "
        f"(ham={int((df.label == 'ham').sum())}, "
        f"spam={int((df.label == 'spam').sum())})"
    )

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean"], df["label"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test  = vectorizer.transform(x_test_text)
    print(f"[prep] vocabulary size = {len(vectorizer.vocabulary_):,}")

    print("\n=== stage 1: spam vs ham ===")
    classifiers = build_classifiers()
    metrics = {}
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        metrics[name] = evaluate(name, y_test, clf.predict(x_test))

    save_pickle("vectorizer.pkl",  vectorizer)
    save_pickle("naive_bayes.pkl", classifiers["Naive Bayes"])
    save_pickle("knn.pkl",         classifiers["KNN"])
    save_pickle("mlp.pkl",         classifiers["MLP"])
    save_pickle("metrics.pkl",     metrics)


def train_categories(df):
    """Stage 2: work / personal / promotion. Uses only rows with a category."""
    df = df[df["category"].isin(CATEGORY_LABELS)].copy()
    if df.empty:
        print("[skip] no category rows found, skipping stage 2.")
        return

    df["clean"] = df["text"].astype(str).apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)
    print(f"[data] stage 2 rows: {len(df):,} ({df['category'].value_counts().to_dict()})")

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean"], df["category"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["category"],
    )

    vectorizer = TfidfVectorizer(
        max_features=2000, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test  = vectorizer.transform(x_test_text)
    print(f"[prep] category vocabulary size = {len(vectorizer.vocabulary_):,}")

    print("\n=== stage 2: work / personal / promotion ===")
    classifiers = build_classifiers()
    for name, clf in classifiers.items():
        clf.fit(x_train, y_train)
        evaluate(name, y_test, clf.predict(x_test), pos_label=None)

    save_pickle("vectorizer_category.pkl",  vectorizer)
    save_pickle("naive_bayes_category.pkl", classifiers["Naive Bayes"])
    save_pickle("knn_category.pkl",         classifiers["KNN"])
    save_pickle("mlp_category.pkl",         classifiers["MLP"])

# Entry point                                                                 

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    download_nltk_data()

    df = pd.read_csv(MAIL_DATASET_PATH, keep_default_na=False)
    print(f"[data] loaded {MAIL_DATASET_PATH} ({len(df):,} rows)")

    train_spam_ham(df)
    train_categories(df)

    print(f"\n[done] all artefacts written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
