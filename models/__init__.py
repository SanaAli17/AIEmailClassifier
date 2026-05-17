"""Models package for the AI Email Classifier."""

# Re-export the model helpers used by the Flask app and scripts.

from .preprocessor import preprocess
from .classifier import (
    EmailClassifier,
    available_models,
    load_model,
    load_vectorizer,
    load_category_model,
    load_category_vectorizer,
    load_metrics,
    load_evaluation,
    category_models_available,
    predict,
    get_top_keywords,
    get_keyword_scores,
    get_sentiment,
)
