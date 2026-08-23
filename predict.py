import joblib
import re
import numpy as np

# Load trained model
model = joblib.load("model/mood_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# Emotion labels
emotion_names = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😠",
    4: "Fear 😨",
    5: "Surprise 😲"
}


# Text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Get input
text = input("How are you feeling? ")

# Clean input
cleaned_text = clean_text(text)

# Convert text to TF-IDF
text_vector = vectorizer.transform([cleaned_text])

# Predict
prediction = model.predict(text_vector)

# Get probabilities
probabilities = model.predict_proba(text_vector)[0]

# Find highest probability
index = np.argmax(probabilities)

# Get emotion
emotion_number = int(model.classes_[index])
emotion = emotion_names.get(emotion_number, "Unknown")

# Confidence
confidence = probabilities[index]


print("\n-----------------------------")
print("Detected Emotion:", emotion)
print("Model Probability:", round(confidence * 100, 2), "%")
print("-----------------------------")