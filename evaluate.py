"""
evaluate.py

Run the training pipeline N times with different random seeds and report
mean ± std for accuracy / precision / recall / F1 on both stages.

Result is saved to trained_models/evaluation.pkl and rendered by the
"Evaluation" tab in the Flask app.

Run:
    python evaluate.py            # 5 runs (default)
    python evaluate.py -n 10      # 10 runs

"""

import argparse
import os

import joblib
import numpy as np
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
EVALUATION_PATH   = os.path.join(MODELS_DIR, "evaluation.pkl")

CATEGORY_LABELS = ("work", "personal", "promotion")
TEST_SIZE       = 0.20


# Evaluation helpers

def make_classifiers(seed):

    """Fresh dict of the three classifiers, with MLP seeded per run."""

    return {
        "Naive Bayes": MultinomialNB(),
        "KNN":         KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "MLP":         MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=30,
            random_state=seed,
        ),
    }


def metrics_for(y_true, y_pred, pos_label):

    # Use the correct precision/recall settings for binary vs multi-class labels.
    if pos_label is None:
        kwargs = {"average": "macro", "zero_division": 0}
    else:
        kwargs = {"pos_label": pos_label, "zero_division": 0}
    return {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, **kwargs),
        "recall":    recall_score(y_true,    y_pred, **kwargs),
        "f1":        f1_score(y_true,        y_pred, **kwargs),
    }


def run_stage(df, label_col, pos_label, max_features, seed):
    """Train all 3 classifiers once at the given seed; return {model: metrics}."""
    df = df.copy()
    df["clean"] = df["text"].astype(str).apply(preprocess)
    df = df[df["clean"].str.len() > 0].reset_index(drop=True)

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["clean"], df[label_col],
        test_size=TEST_SIZE, random_state=seed, stratify=df[label_col],
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(x_train_text)
    x_test  = vectorizer.transform(x_test_text)

    out = {}
    for name, clf in make_classifiers(seed).items():
        clf.fit(x_train, y_train)
        out[name] = metrics_for(y_test, clf.predict(x_test), pos_label)
    return out


def aggregate(runs_list):

    # Collapse repeated runs into mean/std summaries for each metric.
    models  = list(runs_list[0].keys())
    metrics = list(runs_list[0][models[0]].keys())

    result = {}
    for model in models:
        result[model] = {}
        for metric in metrics:
            values = [r[model][metric] for r in runs_list]
            # ddof=1 is sample std (n-1 denominator) the standard choice
            result[model][metric] = {
                "runs": [float(v) for v in values],
                "mean": float(np.mean(values)),
                "std":  float(np.std(values, ddof=1)),
            }
    return result


def print_table(stage_name, agg):
    """Print one stage's aggregated table to the terminal."""
    print(f"\n=== {stage_name} (mean ± std over {len(next(iter(agg.values()))['accuracy']['runs'])} runs) ===")
    print(f"  {'Model':<14}  {'Accuracy':<18}  {'Precision':<18}  {'Recall':<18}  {'F1':<18}")
    for model, metrics in agg.items():
        cells = []
        for key in ("accuracy", "precision", "recall", "f1"):
            m, s = metrics[key]["mean"], metrics[key]["std"]
            cells.append(f"{m:.4f} ± {s:.4f}")
        print(f"  {model:<14}  " + "  ".join(f"{c:<18}" for c in cells))


# Entry point


def main():
    parser = argparse.ArgumentParser(description="Run training N times and report mean ± std.")
    parser.add_argument("-n", "--runs", type=int, default=5, help="number of runs (default 5)")
    args = parser.parse_args()

    if args.runs < 2:
        raise SystemExit("Need at least 2 runs to compute a std.")

    seeds = list(range(42, 42 + args.runs))
    print(f"[eval] {len(seeds)} runs, seeds = {seeds}")

    download_nltk_data()
    df = pd.read_csv(MAIL_DATASET_PATH, keep_default_na=False)
    print(f"[data] loaded {MAIL_DATASET_PATH} ({len(df):,} rows)")

    stage1_df = df[df["label"].isin(["ham", "spam"])]
    stage2_df = df[df["category"].isin(CATEGORY_LABELS)]

    s1_runs, s2_runs = [], []
    for i, seed in enumerate(seeds, 1):
        print(f"\n--- run {i}/{len(seeds)} (seed={seed}) ---")
        s1_runs.append(run_stage(stage1_df, "label",    "spam", 5000, seed))
        s2_runs.append(run_stage(stage2_df, "category", None,   2000, seed))

    result = {
        "n_runs": len(seeds),
        "seeds":  seeds,
        "stage1": aggregate(s1_runs),
        "stage2": aggregate(s2_runs),
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(result, EVALUATION_PATH)

    print_table("stage 1: spam vs ham",                     result["stage1"])
    print_table("stage 2: work / personal / promotion",     result["stage2"])
    print(f"\n[saved] {EVALUATION_PATH}")


if __name__ == "__main__":
    main()
