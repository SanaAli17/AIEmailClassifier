"""
app.py

Flask web app for the AI Email Classifier.

Routes:
    GET  /                 -> input form
    POST /classify         -> run prediction, save to DB, show result
    GET  /history          -> list of past predictions + charts
    POST /clear-history    -> empty the DB, redirect back to /history
    GET  /chart/<type>     -> raw PNG chart (performance/wordfreq/distribution/categories)
    
"""

# Matplotlib must be set to headless mode BEFORE importing pyplot.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import base64
import io
import os
import sqlite3
from collections import Counter
from datetime import datetime

import numpy as np
from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from models import (
    available_models,
    category_models_available,
    get_keyword_scores,
    get_sentiment,
    get_top_keywords,
    load_category_model,
    load_category_vectorizer,
    load_evaluation,
    load_metrics,
    load_model,
    load_vectorizer,
    predict,
    preprocess,
)

# Paths and constants                                                         

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR  = os.path.join(BASE_DIR, "database")
DATABASE_PATH = os.path.join(DATABASE_DIR, "history.db")
MAX_UPLOAD_MB = 2          # max .txt upload size
SNIPPET_LEN   = 120        # how many chars of email to keep as a preview

# Database: simple sqlite helpers                                             

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email_snippet TEXT NOT NULL,
    full_text     TEXT NOT NULL,
    label         TEXT NOT NULL,
    confidence    REAL NOT NULL,
    sentiment     TEXT NOT NULL,
    model_used    TEXT NOT NULL,
    timestamp     TEXT NOT NULL,
    category      TEXT
)
"""


def db_connect():
    """Open a SQLite connection. Rows behave like dicts."""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the predictions table the first time the app runs."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    conn = db_connect()
    conn.execute(SCHEMA)
    # Add `category` column if upgrading from an older schema.
    columns = {row[1] for row in conn.execute("PRAGMA table_info(predictions)")}
    if "category" not in columns:
        conn.execute("ALTER TABLE predictions ADD COLUMN category TEXT")
    conn.commit()
    conn.close()


def save_prediction(snippet, full_text, label, confidence, sentiment, model_used, category):
    """Insert one prediction row. Timestamp is set here (ISO format)."""
    conn = db_connect()
    conn.execute(
        """INSERT INTO predictions
           (email_snippet, full_text, label, confidence, sentiment, model_used, timestamp, category)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            snippet,
            full_text,
            label,
            float(confidence),
            sentiment,
            model_used,
            datetime.now().isoformat(timespec="seconds"),
            category,
        ),
    )
    conn.commit()
    conn.close()


def get_all_records():
    """All saved predictions, newest first, as a list of dicts."""
    conn = db_connect()
    rows = conn.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_label_counts():
    """Return {'spam': N, 'ham': M} from the DB."""
    conn = db_connect()
    rows = conn.execute("SELECT label, COUNT(*) AS n FROM predictions GROUP BY label").fetchall()
    conn.close()
    return {r["label"]: r["n"] for r in rows}


def get_category_counts():
    """Return {'work': N, 'personal': M, 'promotion': K} from the DB."""
    conn = db_connect()
    rows = conn.execute(
        "SELECT category, COUNT(*) AS n FROM predictions "
        "WHERE category IS NOT NULL GROUP BY category"
    ).fetchall()
    conn.close()
    return {r["category"]: r["n"] for r in rows}


def get_all_full_texts():
    """All full email texts. Used by the /chart/wordfreq endpoint."""
    conn = db_connect()
    rows = conn.execute("SELECT full_text FROM predictions").fetchall()
    conn.close()
    return [r["full_text"] for r in rows]


def clear_predictions():
    """Empty the predictions table."""
    conn = db_connect()
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()


# Chart helpers (Matplotlib)                                     

COLOR_PRIMARY   = "#4f46e5"   # indigo
COLOR_SECONDARY = "#10b981"   # green
COLOR_ACCENT    = "#f59e0b"   # amber
COLOR_DANGER    = "#ef4444"   # red
COLOR_NEUTRAL   = "#6b7280"   # gray


def fig_to_png(fig):
    """Save a matplotlib figure to a PNG byte string and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def to_data_uri(png_bytes):
    """Wrap raw PNG bytes as a `data:image/png;base64,...` string for <img>."""
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def placeholder_chart(message):
    """Empty chart with a friendly message in the middle."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color=COLOR_NEUTRAL)
    ax.axis("off")
    return fig_to_png(fig)


def chart_performance(metrics):
    """Grouped bar chart: accuracy / precision / recall / F1 per model."""
    if not metrics:
        return placeholder_chart("No metrics yet, run train.py")

    models = list(metrics.keys())
    keys   = ["accuracy", "precision", "recall", "f1"]
    values = np.array([[metrics[m][k] for k in keys] for m in models])
    colors = [COLOR_PRIMARY, COLOR_SECONDARY, COLOR_ACCENT, COLOR_DANGER]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    width = 0.2
    for i, key in enumerate(keys):
        ax.bar(x + i * width - 1.5 * width, values[:, i], width=width,
               label=key.capitalize(), color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    return fig_to_png(fig)


def chart_word_frequency(keywords):
    """Horizontal bar chart: top keywords with their TF-IDF scores."""
    if not keywords:
        return placeholder_chart("No in-vocab keywords")

    labels = [k for k, _ in keywords][::-1]   # reverse so biggest is on top
    scores = [s for _, s in keywords][::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, scores, color=COLOR_PRIMARY)
    ax.set_xlabel("Score")
    ax.set_title("Top Keywords")
    ax.grid(axis="x", alpha=0.25)
    return fig_to_png(fig)


def chart_distribution(label_counts):
    """Pie chart: spam vs ham."""
    if not label_counts:
        return placeholder_chart("No predictions yet")

    labels = list(label_counts.keys())
    sizes  = [label_counts[l] for l in labels]
    color_map = {"spam": COLOR_DANGER, "ham": COLOR_SECONDARY}
    colors = [color_map.get(l, COLOR_NEUTRAL) for l in labels]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        sizes,
        labels=[l.capitalize() for l in labels],
        colors=colors,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"linewidth": 2, "edgecolor": "white"},
    )
    ax.set_title("Spam vs. Ham Distribution")
    return fig_to_png(fig)


def chart_categories(category_counts):
    """Bar chart: work / personal / promotion counts (ham predictions only)."""
    if not category_counts:
        return placeholder_chart("No ham predictions yet")

    # Show the categories in a fixed order so the bars don't jump around.
    fixed_order = ["work", "personal", "promotion"]
    labels = [l for l in fixed_order if l in category_counts]
    labels += [l for l in category_counts if l not in fixed_order]
    sizes  = [category_counts[l] for l in labels]
    color_map = {"work": COLOR_PRIMARY, "personal": COLOR_ACCENT, "promotion": COLOR_SECONDARY}
    colors = [color_map.get(l, COLOR_NEUTRAL) for l in labels]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        [l.capitalize() for l in labels], sizes,
        color=colors, edgecolor="white", linewidth=1.5,
    )
    ax.set_ylim(0, max(sizes) * 1.15 + 1)
    ax.set_ylabel("Predictions")
    ax.set_title("Ham Category Distribution")
    ax.grid(axis="y", alpha=0.25)
    for bar, n in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                str(n), ha="center", va="bottom", fontsize=10)
    return fig_to_png(fig)


# Model cache                                                                 

# Models are big; load each one only the first time it's requested, then keep
# it in memory.
_MODEL_CACHE = {}
_CATEGORY_CACHE = {}
_VECTORIZER = None
_CATEGORY_VECTORIZER = None


def get_model(model_name):
    """Load and cache a stage-1 (spam/ham) model."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = load_model(model_name)
    return _MODEL_CACHE[model_name]


def get_vectorizer():
    """Load and cache the stage-1 TF-IDF vectorizer."""
    global _VECTORIZER
    if _VECTORIZER is None:
        _VECTORIZER = load_vectorizer()
    return _VECTORIZER


def get_category_model(model_name):
    """Load and cache a stage-2 (work/personal/promotion) model. None if missing."""
    if not category_models_available():
        return None
    if model_name not in _CATEGORY_CACHE:
        _CATEGORY_CACHE[model_name] = load_category_model(model_name)
    return _CATEGORY_CACHE[model_name]


def get_category_vectorizer():
    """Load and cache the stage-2 TF-IDF vectorizer. None if missing."""
    global _CATEGORY_VECTORIZER
    if not category_models_available():
        return None
    if _CATEGORY_VECTORIZER is None:
        _CATEGORY_VECTORIZER = load_category_vectorizer()
    return _CATEGORY_VECTORIZER


# Small request helpers                                                       

def read_email_text():
    """Pull email text from the form. Textarea wins, otherwise .txt upload."""
    text = (request.form.get("email_text") or "").strip()
    if text:
        return text

    uploaded = request.files.get("email_file")
    if uploaded and uploaded.filename:
        if not uploaded.filename.lower().endswith(".txt"):
            abort(400, "Only .txt files are accepted.")
        return uploaded.read().decode("utf-8", errors="replace").strip()

    return ""


def aggregate_keywords(top_n=15):
    """Most common cleaned tokens across the entire history (for /chart/wordfreq)."""
    texts = get_all_full_texts()
    if not texts:
        return []
    bag = Counter()
    for text in texts:
        for token in preprocess(text).split():
            bag[token] += 1
    return [(word, float(count)) for word, count in bag.most_common(top_n)]


# Flask app + routes                                                          

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Run DB setup once when the module is imported.
init_db()


@app.route("/", methods=["GET"])
def index():
    """Landing page with the email-input form."""
    return render_template("index.html", models=available_models())


@app.route("/classify", methods=["POST"])
def classify():
    """Run preprocessor + model, save to DB, render result page."""
    text = read_email_text()
    if not text:
        abort(400, "Please paste email text or upload a .txt file.")

    model_name = request.form.get("model", "Naive Bayes")
    if model_name not in available_models():
        abort(400, f"Unknown model: {model_name}")

    # ---- Run stage 1: spam vs ham --------------------------------------- #
    model      = get_model(model_name)
    vectorizer = get_vectorizer()
    result     = predict(model, vectorizer, text, model_name)

    keywords        = get_top_keywords(vectorizer, text, top_n=10)
    keyword_scores  = get_keyword_scores(vectorizer, text, top_n=10)
    sentiment_label = get_sentiment(text)

    # ---- Run stage 2: ham -> work / personal / promotion ---------------- #
    category            = None
    category_confidence = None
    if result["label"] == "ham":
        cat_model = get_category_model(model_name)
        cat_vec   = get_category_vectorizer()
        if cat_model is not None and cat_vec is not None:
            cat_result          = predict(cat_model, cat_vec, text, model_name)
            category            = cat_result["label"]
            category_confidence = float(cat_result["confidence"])

    # ---- Save to history database --------------------------------------- #
    snippet = text[:SNIPPET_LEN] + ("…" if len(text) > SNIPPET_LEN else "")
    save_prediction(
        snippet=snippet,
        full_text=text,
        label=result["label"],
        confidence=result["confidence"],
        sentiment=sentiment_label,
        model_used=result["model"],
        category=category,
    )

    # ---- Build all four charts as base64 data URIs ---------------------- #
    metrics_data = load_metrics()
    chart_perf   = to_data_uri(chart_performance(metrics_data))
    chart_words  = to_data_uri(chart_word_frequency(keyword_scores))
    chart_dist   = to_data_uri(chart_distribution(get_label_counts()))
    chart_cats   = to_data_uri(chart_categories(get_category_counts()))

    return render_template(
        "result.html",
        label=result["label"],
        confidence=float(result["confidence"]),
        confidence_pct=round(float(result["confidence"]) * 100, 1),
        model=result["model"],
        sentiment=sentiment_label,
        keywords=keywords,
        email_snippet=snippet,
        email_text=text,
        available_models=available_models(),
        category=category,
        category_confidence_pct=(
            round(category_confidence * 100, 1)
            if category_confidence is not None else None
        ),
        chart_performance=chart_perf,
        chart_wordfreq=chart_words,
        chart_distribution=chart_dist,
        chart_categories=chart_cats,
    )


@app.route("/history", methods=["GET"])
def history():
    """Show the table of past predictions plus the two summary charts."""
    records   = get_all_records()
    chart_dist = to_data_uri(chart_distribution(get_label_counts()))
    chart_cats = to_data_uri(chart_categories(get_category_counts()))
    return render_template(
        "history.html",
        records=records,
        chart_distribution=chart_dist,
        chart_categories=chart_cats,
        total=len(records),
    )


@app.route("/evaluation", methods=["GET"])
def evaluation():
    """Show the mean ± std evaluation table."""
    return render_template("evaluation.html", data=load_evaluation())


@app.route("/clear-history", methods=["POST"])
def clear_history():
    """Empty the predictions table and go back to /history."""
    clear_predictions()
    return redirect(url_for("history"))


@app.route("/chart/<chart_type>", methods=["GET"])
def chart(chart_type):
    """Return a raw PNG for any of the four chart types. Useful for debugging."""
    if chart_type == "performance":
        png = chart_performance(load_metrics())
    elif chart_type == "wordfreq":
        png = chart_word_frequency(aggregate_keywords())
    elif chart_type == "distribution":
        png = chart_distribution(get_label_counts())
    elif chart_type == "categories":
        png = chart_categories(get_category_counts())
    else:
        abort(404, f"Unknown chart type: {chart_type}")
    return send_file(io.BytesIO(png), mimetype="image/png")


# Entry point                                                                 

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
