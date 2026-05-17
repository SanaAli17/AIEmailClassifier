"""
preprocessor.py

Cleans email text before TF-IDF.

Steps (in order):
    1. Strip HTML tags  (BeautifulSoup)
    2. Lowercase
    3. Tokenize         (NLTK word_tokenize)
    4. Keep only alphabetic tokens of length >= 2
    5. Remove English stopwords
    6. Lemmatize        
    7. Join tokens back into a single string

The same function is used in train.py and app.py so the cleaned
text looks identical at training time and prediction time.
"""

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# NLTK datasets we need. Downloaded once, then cached on disk.
NLTK_PACKAGES = [
    ("tokenizers/punkt",     "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords",    "stopwords"),
    ("corpora/wordnet",      "wordnet"),
    ("corpora/omw-1.4",      "omw-1.4"),
]


def download_nltk_data():
    """Download every NLTK resource we need, if not already present."""
    for path, package in NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


# Download once at import time so later calls are fast.
download_nltk_data()

# Build these once at module load.
STOPWORDS  = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()
TOKEN_RE   = re.compile(r"^[a-z]{2,}$")   # only letters, 2+ chars


def preprocess(text):
    """Run all cleaning steps and return a single space-separated string."""
    if not text:
        return ""

    # 1. Strip HTML tags
    text = BeautifulSoup(str(text), "html.parser").get_text(separator=" ")

    # 2. Lowercase
    text = text.lower()

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Keep only alphabetic tokens (length >= 2)
    tokens = [t for t in tokens if TOKEN_RE.match(t)]

    # 5. Remove stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    # 6. Lemmatize
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]

    # 7. Join back into one string
    return " ".join(tokens)
