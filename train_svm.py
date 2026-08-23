import pandas as pd
import re
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


# --------------------------------
# Load Dataset
# --------------------------------

train = pd.read_parquet("dataset/train.parquet")


# --------------------------------
# Text Cleaning
# --------------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train["text"] = train["text"].apply(clean_text)


# --------------------------------
# Input and Labels
# --------------------------------

X_train = train["text"]
y_train = train["label"]


# --------------------------------
# TF-IDF
# --------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)


print("TF-IDF completed!")


# --------------------------------
# Calibrated Linear SVM
# --------------------------------

svm = LinearSVC(
    C=1.5,
    class_weight="balanced"
)

model = CalibratedClassifierCV(
    svm,
    cv=3
)

model.fit(
    X_train_tfidf,
    y_train
)

print("SVM training completed!")


# --------------------------------
# Save Model
# --------------------------------

os.makedirs("model", exist_ok=True)

joblib.dump(
    model,
    "model/svm_mood_model.pkl"
)

joblib.dump(
    vectorizer,
    "model/svm_tfidf_vectorizer.pkl"
)

print("\nSVM model saved successfully!")
print("Vectorizer saved successfully!")