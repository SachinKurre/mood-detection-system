import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer

# Load datasets
train = pd.read_parquet("dataset/train.parquet")
test = pd.read_parquet("dataset/test.parquet")

print("Training samples:", len(train))
print("Testing samples:", len(test))


# -----------------------------
# Text cleaning
# -----------------------------

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


train["text"] = train["text"].apply(clean_text)
test["text"] = test["text"].apply(clean_text)


# -----------------------------
# Separate input and labels
# -----------------------------

X_train = train["text"]
y_train = train["label"]

X_test = test["text"]
y_test = test["label"]


# -----------------------------
# TF-IDF
# -----------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF completed!")

print("Training feature shape:", X_train_tfidf.shape)
print("Testing feature shape:", X_test_tfidf.shape)

from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")

# -----------------------------
# Model Evaluation
# -----------------------------

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Make predictions
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(round(accuracy * 100, 2), "%")

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted Emotion")
plt.ylabel("Actual Emotion")
plt.title("Emotion Detection Confusion Matrix")

plt.show()

import os
import joblib

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model
joblib.dump(model, "model/mood_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")