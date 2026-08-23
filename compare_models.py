import pandas as pd
import re
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------
# Load Dataset
# --------------------------------

train = pd.read_parquet("dataset/train.parquet")
test = pd.read_parquet("dataset/test.parquet")


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
test["text"] = test["text"].apply(clean_text)


# --------------------------------
# Separate Input and Labels
# --------------------------------

X_train = train["text"]
y_train = train["label"]

X_test = test["text"]
y_test = test["label"]


# --------------------------------
# TF-IDF
# --------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# =================================
# MODEL 1: LOGISTIC REGRESSION
# =================================

logistic_model = LogisticRegression(
    max_iter=1000
)

logistic_model.fit(
    X_train_tfidf,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_tfidf
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)


# =================================
# MODEL 2: LINEAR SVM
# =================================

svm_model = LinearSVC(
    class_weight="balanced"
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_prediction = svm_model.predict(
    X_test_tfidf
)

svm_accuracy = accuracy_score(
    y_test,
    svm_prediction
)


# =================================
# RESULTS
# =================================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print(
    f"Logistic Regression: "
    f"{logistic_accuracy * 100:.2f}%"
)

print(
    f"Linear SVM: "
    f"{svm_accuracy * 100:.2f}%"
)


# --------------------------------
# Select Best Model
# --------------------------------

if svm_accuracy > logistic_accuracy:

    print("\n🏆 Best Model: Linear SVM")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            svm_prediction
        )
    )

else:

    print("\n🏆 Best Model: Logistic Regression")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            logistic_prediction
        )
    )
    # --------------------------------
    # Save Best Model
    # --------------------------------

    os.makedirs("model", exist_ok=True)

    if svm_accuracy > logistic_accuracy:

        joblib.dump(
            svm_model,
            "model/svm_mood_model.pkl"
        )

        joblib.dump(
            vectorizer,
            "model/svm_tfidf_vectorizer.pkl"
        )

        print("\nSVM model and vectorizer saved successfully!")

    else:

        joblib.dump(
            logistic_model,
            "model/mood_model.pkl"
        )

        joblib.dump(
            vectorizer,
            "model/tfidf_vectorizer.pkl"
        )

        print("\nLogistic Regression model and vectorizer saved successfully!")