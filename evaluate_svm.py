import pandas as pd
import re
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------
# Load Test Dataset
# --------------------------------

test = pd.read_parquet(
    "dataset/test.parquet"
)


# --------------------------------
# Text Cleaning
# --------------------------------

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


test["text"] = test["text"].apply(
    clean_text
)


# --------------------------------
# Input and Labels
# --------------------------------

X_test = test["text"]
y_test = test["label"]


# --------------------------------
# Load SVM Model
# --------------------------------

model = joblib.load(
    "model/svm_mood_model.pkl"
)

vectorizer = joblib.load(
    "model/svm_tfidf_vectorizer.pkl"
)


# --------------------------------
# Convert Text to TF-IDF
# --------------------------------

X_test_tfidf = vectorizer.transform(
    X_test
)


# --------------------------------
# Prediction
# --------------------------------

y_pred = model.predict(
    X_test_tfidf
)


# --------------------------------
# Accuracy
# --------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("FINAL SVM MODEL EVALUATION")
print("==============================")

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# --------------------------------
# Classification Report
# --------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------
# Confusion Matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\nConfusion Matrix:")

print(cm)


# --------------------------------
# Confusion Matrix Graph
# --------------------------------

labels = [
    "Sadness",
    "Joy",
    "Love",
    "Anger",
    "Fear",
    "Surprise"
]

plt.figure(
    figsize=(8, 6)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.xlabel(
    "Predicted Emotion"
)

plt.ylabel(
    "Actual Emotion"
)

plt.title(
    "SVM Emotion Detection Confusion Matrix"
)

plt.tight_layout()

plt.tight_layout()

plt.savefig(
    "model/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()