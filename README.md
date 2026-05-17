# AI Email Classifier

A small web app that decides whether an email is **spam** or **ham** (a normal email).
It uses three classic machine-learning models from scikit-learn and runs on a
Flask backend with simple HTML pages.

## What It Includes

- Web GUI with four tabs: Classify, History, Evaluation, plus the per-email Result page.
- Email text input and `.txt` file upload.
- Preprocessing pipeline:
  - HTML tag removal (BeautifulSoup)
  - lowercasing
  - tokenization (NLTK)
  - stopword removal
  - lemmatization (WordNet)
- TF-IDF vectorization (uni- and bi-grams, 5,000 features for stage 1, 2,000 for stage 2).
- Three trained models per stage:
  - Multinomial Naive Bayes
  - K-Nearest Neighbors (k = 5, cosine distance)
  - Multi-Layer Perceptron (two hidden layers, 128 -> 64 ReLU units)
- Spam / Not Spam prediction with confidence score.
- A second-stage classifier that labels ham emails as **work**, **personal**, or **promotion**.
- VADER sentiment (positive / negative / neutral).
- Top 10 keywords from the email (ranked by TF-IDF).
- Four charts: model performance, word frequency, spam vs ham distribution, ham category distribution.
- Persistent prediction history in SQLite with a clear-history button.
- Evaluation tab: mean ± standard deviation table over N training runs.

## How To Run (from scratch)

From inside the `mail-sense/` folder:

```
# 1. Create a virtual environment (recommended on macOS Homebrew Python)
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the models (writes 9 .pkl files into trained_models/)
python train.py

# 4. Run the 5-run evaluation (writes trained_models/evaluation.pkl)
python evaluate.py                  # or:  python evaluate.py -n 10

# 5. Start the web app
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

Tabs: **Classify** (main form), **History** (past predictions),
**Evaluation** (mean ± std table from `evaluate.py`).

If you skip the venv, plain `pip install` will fail on Homebrew Python with
`externally-managed-environment`. In that case use:
`pip3 install -r requirements.txt --user --break-system-packages`.

## Dataset

The project reads ONE file:

```
database/mail_dataset.csv
```

It has three columns:

- `text` — the email body
- `label` — `spam` or `ham`
- `category` — `work`, `personal`, `promotion`, or empty

34,058 rows total:

- **Enron-Spam corpus** (Metsis, Androutsopoulos & Paliouras, CEAS 2006) —
  33,818 rows labelled `ham` / `spam`.
- **Short hand-labelled emails** — 240 rows (80 each) tagged with a
  `category` of `work` / `personal` / `promotion`. These are also labelled `ham`
  so stage 1 treats them as legitimate mail.

`train.py` and `evaluate.py` both read this CSV directly with `pd.read_csv()`.
The Flask app does **not** read the CSV at runtime — it only loads the trained
`.pkl` files from `trained_models/`.

### Importing the Kaggle Enron Spam Data

The Kaggle dataset from Marcel Wiechmann uses different column names:

```
Subject, Message, Spam/Ham, Date
```

To replace the spam/ham rows with that dataset while keeping the local
`work` / `personal` / `promotion` category rows, run:

```
python import_kaggle_dataset.py
```

The script reads the dataset from the public GitHub mirror. You can also pass a
downloaded CSV or ZIP file:

```
python import_kaggle_dataset.py path/to/enron_spam_data.csv
```

The script writes `database/mail_dataset.csv` and keeps the existing
category-labelled rows so stage 2 still works. It does not keep the downloaded
ZIP because the active CSV already contains the imported data.

After importing, rebuild the generated model files:

```
python train.py
python evaluate.py
```

## Project Layout

```
mail-sense/
├── app.py                  # Flask routes, SQLite + chart helpers
├── train.py                # Trains 6 models, writes .pkl files
├── evaluate.py             # 5-run mean ± std evaluation
├── requirements.txt
│
├── models/
│   ├── preprocessor.py     # Text cleaning pipeline
│   └── classifier.py       # Load .pkl + predict / keywords / sentiment
│
├── trained_models/         # Output of train.py / evaluate.py (gitignored)
│   ├── vectorizer.pkl
│   ├── naive_bayes.pkl, knn.pkl, mlp.pkl, metrics.pkl
│   ├── vectorizer_category.pkl
│   ├── naive_bayes_category.pkl, knn_category.pkl, mlp_category.pkl
│   └── evaluation.pkl
│
├── templates/              # Jinja2 HTML
│   ├── index.html, result.html, history.html, evaluation.html
│
├── static/style.css
│
└── database/
    ├── mail_dataset.csv    # The single dataset file
    └── history.db          # SQLite predictions (created at runtime)
```

## Project Flow

```
User Input -> Preprocessing -> TF-IDF -> Trained Model -> Prediction -> GUI + Charts
```

## Notes

- VADER sentiment runs on the raw text (before cleaning) so casing and punctuation still count.
- The TF-IDF vectorizer is fit only on the training split, not the test split.
- The history table is a tiny SQLite file (`database/history.db`); deleting it
  resets the history without touching the trained models.
- If port 5000 is busy (macOS AirPlay Receiver uses it), edit the last line of
  `app.py` and change `port=5000` to e.g. `port=5001`.
- The `.pkl` files are Python-version sensitive. If a collaborator on a
  different Python version hits errors loading them, have them re-run
  `python train.py`.
