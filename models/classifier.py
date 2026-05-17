"""
classifier.py

Loads the trained .pkl files and runs predictions.

Two stages:
    Stage 1 (spam vs ham):           load_model() + load_vectorizer()
    Stage 2 (work/personal/promo):   load_category_model() + load_category_vectorizer()

All trained files live in trained_models/. we need to run `python train.py` once
before using these functions.

"""

import math
import os
import joblib
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .preprocessor import preprocess


# Paths and model names                                                       

HERE        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(os.path.dirname(HERE), "trained_models")

# Map  model name  on-disk filename for stage 1 (spam vs ham).
MODEL_FILES = {
    "Naive Bayes": "naive_bayes.pkl",
    "KNN":         "knn.pkl",
    "MLP":         "mlp.pkl",
}

# Same map for stage 2 (work / personal / promotion).
CATEGORY_FILES = {
    "Naive Bayes": "naive_bayes_category.pkl",
    "KNN":         "knn_category.pkl",
    "MLP":         "mlp_category.pkl",
}

VECTORIZER_FILE          = "vectorizer.pkl"
CATEGORY_VECTORIZER_FILE = "vectorizer_category.pkl"
METRICS_FILE             = "metrics.pkl"
EVALUATION_FILE          = "evaluation.pkl"

# Reuse the same VADER analyzer for every call.
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


# Loading                                                                     

def available_models():
    """Names shown in the model dropdown on the UI."""
    return list(MODEL_FILES.keys())


def load_pickle(filename):
    """Load any pickle from trained_models/, with a helpful error if missing."""
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run `python train.py` first."
        )
    return joblib.load(path)


def load_model(model_name):
    """Load the stage-1 (spam/ham) model by display name."""
    if model_name not in MODEL_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    return load_pickle(MODEL_FILES[model_name])


def load_vectorizer():
    """Load the stage-1 TF-IDF vectorizer."""
    return load_pickle(VECTORIZER_FILE)


def load_category_model(model_name):
    """Load the stage-2 (work/personal/promotion) model by display name."""
    if model_name not in CATEGORY_FILES:
        raise ValueError(f"Unknown model: {model_name}")
    return load_pickle(CATEGORY_FILES[model_name])


def load_category_vectorizer():
    """Load the stage-2 TF-IDF vectorizer."""
    return load_pickle(CATEGORY_VECTORIZER_FILE)


def load_metrics():
    """Read the stage-1 metrics dict saved by train.py. Empty if missing."""
    path = os.path.join(MODELS_DIR, METRICS_FILE)
    if not os.path.exists(path):
        return {}
    return joblib.load(path)


def load_evaluation():
    """Read mean ± std results saved by evaluate.py. None if missing."""
    path = os.path.join(MODELS_DIR, EVALUATION_FILE)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def category_models_available():
    """True only if every stage-2 file exists on disk."""
    needed = list(CATEGORY_FILES.values()) + [CATEGORY_VECTORIZER_FILE]
    return all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in needed)


# Inference                                                                   

class EmailClassifier:
    """Small OOP wrapper around a trained model and vectorizer."""

    def __init__(self, model, vectorizer, model_name):
        self.model = model
        self.vectorizer = vectorizer
        self.model_name = model_name

    @classmethod
    def from_name(cls, model_name):
        """Create a classifier by loading the saved model and vectorizer."""
        return cls(load_model(model_name), load_vectorizer(), model_name)

    @classmethod
    def category_from_name(cls, model_name):
        """Create a category classifier by loading the saved stage-2 files."""
        return cls(
            load_category_model(model_name),
            load_category_vectorizer(),
            model_name,
        )

    def predict(self, text):
        """Classify one email. Returns {'label', 'confidence', 'model'}."""
        return predict(self.model, self.vectorizer, text, self.model_name)

    def get_top_keywords(self, text, top_n=10):
        """Top-N highest-TF-IDF tokens that appear in this email."""
        return get_top_keywords(self.vectorizer, text, top_n)

    def get_keyword_scores(self, text, top_n=10):
        """Return (word, score) pairs for charting."""
        return get_keyword_scores(self.vectorizer, text, top_n)


def _confidence(model, X):
    """Return a number in [0, 1] for the predicted class.

    Most sklearn models have predict_proba(). KNN and MLP do; Naive Bayes does
    too. We use a sigmoid fallback for the rare case a model only has
    decision_function().
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        return float(np.max(probabilities))

    if hasattr(model, "decision_function"):
        scores = np.atleast_1d(model.decision_function(X)[0])
        best = float(np.max(scores))
        return float(1.0 / (1.0 + math.exp(-abs(best))))

    return 1.0


def predict(model, vectorizer, text, model_name):
    """Classify one email. Returns {'label', 'confidence', 'model'}."""
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    label = str(model.predict(X)[0])
    return {
        "label":      label,
        "confidence": _confidence(model, X),
        "model":      model_name,
    }


def get_top_keywords(vectorizer, text, top_n=10):
    """Top-N highest-TF-IDF tokens that appear in this email."""
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    if X.nnz == 0:
        return []

    scores = X.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    # argsort sorts ascending, so reverse and take the top_n.
    ranked = np.argsort(scores)[::-1][:top_n]
    return [str(feature_names[i]) for i in ranked if scores[i] > 0]


def get_keyword_scores(vectorizer, text, top_n=10):
    """Same as get_top_keywords but returns (word, score) pairs for charts."""
    cleaned = preprocess(text)
    X = vectorizer.transform([cleaned])
    if X.nnz == 0:
        return []

    scores = X.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    ranked = np.argsort(scores)[::-1][:top_n]
    return [(str(feature_names[i]), float(scores[i])) for i in ranked if scores[i] > 0]


def get_sentiment(text):
    """Return 'positive', 'negative' or 'neutral' using VADER."""
    if not text:
        return "neutral"
    compound = SENTIMENT_ANALYZER.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"
