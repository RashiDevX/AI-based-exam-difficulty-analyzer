import os
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- STREAMLIT CLOUD SAFE NLTK SETUP ----------------

# Force NLTK data to a writable directory
NLTK_DATA_DIR = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

def ensure_nltk_resources():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]

    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, download_dir=NLTK_DATA_DIR, quiet=True)

ensure_nltk_resources()

# ---------------- PREPROCESS SETUP ----------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------- TEXT PREPROCESSING ----------------
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)
