import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from preprocess import preprocess_text

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/final_questions.csv")
df = df.drop_duplicates(subset="question")

df["question"] = df["question"].apply(preprocess_text)

X = df["question"]
y = df["difficulty"]

# ---------------- SPLIT DATA ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- BUILD MODEL ----------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

# ---------------- TRAIN ----------------
model.fit(X_train, y_train)

# ---------------- PREDICT ----------------
y_pred = model.predict(X_test)

# ---------------- METRICS ----------------
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ---------------- CONFUSION MATRIX ----------------
labels = ["Easy", "Medium", "Hard"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(cm)

# ---------------- CONFUSION MATRIX PLOT ----------------
# plt.figure(figsize=(6, 5))
# sns.heatmap(
#     cm,
#     annot=True,
#     fmt="d",
#     cmap="Blues",
#     xticklabels=labels,
#     yticklabels=labels
# )
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix – Question Difficulty Classifier")
# plt.tight_layout()
# plt.show()

# ---------------- SAVE MODEL ----------------
with open("model/difficulty_model.pkl", "wb") as f:
    pickle.dump(model, f)
