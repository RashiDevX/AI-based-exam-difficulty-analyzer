import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- ENSURE NLTK DATA (SAFE FOR CLOUD) ----------------
def ensure_nltk_resources():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

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
