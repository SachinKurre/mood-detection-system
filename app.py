import streamlit as st
import joblib
import re
import pandas as pd
import os

# -----------------------------
# Persistent Mood History
# -----------------------------

history_file = "data/mood_history.csv"

if os.path.exists(history_file):
    st.session_state.history = pd.read_csv(
        history_file
    ).to_dict("records")
else:
    st.session_state.history = []

    
# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("model/svm_mood_model.pkl")
vectorizer = joblib.load("model/svm_tfidf_vectorizer.pkl")


# -----------------------------
# Emotion Labels
# -----------------------------

emotion_labels = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😡",
    4: "Fear 😨",
    5: "Surprise 😲"
}


# -----------------------------
# Text Cleaning
# -----------------------------

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


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="AI Mood Detector",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# Sidebar Navigation
# -----------------------------

st.sidebar.title("🧠 AI Mood Detector")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "ℹ️ About Model"
    ]
)

if page == "🏠 Home":

    # -----------------------------
    # Header
    # -----------------------------

    st.title("🧠 AI Mood Detector")

    st.write(
        "### Understand the emotion behind your text"
    )

    st.write(
        "Enter a sentence describing how you feel, "
        "and our machine learning model will predict "
        "the emotion expressed in the text."
    )

    st.divider()


    # -----------------------------
    # Text Input
    # -----------------------------

    user_text = st.text_area(
        "💬 How are you feeling?",
        placeholder=(
            "Example: I am really happy because "
            "I passed my examination!"
        ),
        height=150
    )


    # -----------------------------
    # Prediction Button
    # -----------------------------

    if st.button(
        "🔍 Detect My Mood",
        use_container_width=True
    ):

        if not user_text.strip():

            st.warning(
                "⚠️ Please enter some text first."
            )

        else:

            # Clean text

            cleaned_text = clean_text(
                user_text
            )

            # TF-IDF

            text_vector = vectorizer.transform(
                [cleaned_text]
            )

            # -----------------------------
            # Prediction
            # -----------------------------

            prediction = model.predict(
                text_vector
            )[0]


            # -----------------------------
            # Probabilities
            # -----------------------------

            probabilities = model.predict_proba(
                text_vector
            )[0]

            confidence = max(
                probabilities
            ) * 100
        


            # -----------------------------
            # Confidence Level
            # -----------------------------

            if confidence >= 70:
                confidence_message = "High confidence ✅"

            elif confidence >= 50:
                confidence_message = "Moderate confidence ⚠️"

            else:
                confidence_message = "Low confidence ⚠️"

            # Emotion

            emotion = emotion_labels[
                prediction
            ]

            # Save prediction to history
            st.session_state.history.append({
                "Text": user_text,
                "Mood": emotion,
                "Confidence": round(confidence, 2)
            })

            # Save history to CSV
            history_df = pd.DataFrame(
                st.session_state.history
            )

            history_df.to_csv(
                history_file,
                index=False
            )


            # -----------------------------
            # Result
            # -----------------------------

            st.divider()

            st.subheader(
                "🎯 Detected Mood"
            )

            st.markdown(
                f"# {emotion}"
            )

            st.progress(
                int(confidence)
            )

            st.write(
                f"**Confidence: {confidence:.2f}%**"
            )

            st.write(
                f"**{confidence_message}**"
            )

            if confidence < 50:
                st.warning(
                    "The model is not very confident about this "
                    "prediction. Try entering a longer sentence "
                    "that describes your feelings more clearly."
                )


            # -----------------------------
            # Probability Table
            # -----------------------------

            st.subheader(
                "📊 Emotion Probabilities"
            )

            probability_data = []

            for i, probability in enumerate(
                probabilities
            ):

                probability_data.append({
                    "Emotion":
                        emotion_labels[i],

                    "Probability":
                        round(
                            probability * 100,
                            2
                        )
                })


            df = pd.DataFrame(
                probability_data
            )

            st.bar_chart(
                df.set_index("Emotion")
            )


            # -----------------------------
            # User Input
            # -----------------------------

            st.subheader(
                "📝 Your Text"
            )

            st.info(user_text)


            # -----------------------------
            # Disclaimer
            # -----------------------------

            st.warning(
                "⚠️ This system detects emotions "
                "from text using machine learning. "
                "It is not a medical or psychological "
                "diagnosis."
            )


    # -----------------------------
    # Footer
    # -----------------------------

    st.divider()

    st.caption(
        "AI Mood Detector • Machine Learning Project"
    )

# -----------------------------
# Dashboard Page
# -----------------------------

elif page == "📊 Dashboard":

    st.title("📊 Mood Dashboard")

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(
            st.session_state.history
        )

        # -----------------------------
        # Dashboard Statistics
        # -----------------------------

        total_predictions = len(history_df)

        most_common_mood = history_df["Mood"].mode()[0]

        average_confidence = history_df["Confidence"].mean()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "🔢 Total Predictions",
                total_predictions
            )

        with col2:
            st.metric(
                "😊 Most Common Mood",
                most_common_mood
            )

        with col3:
            st.metric(
                "🎯 Average Confidence",
                f"{average_confidence:.2f}%"
            )

        st.subheader("📝 Prediction History")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        # -----------------------------
        # Download History
        # -----------------------------

        csv_data = history_df.to_csv(
            index=False
        )

        st.download_button(
            label="📥 Download Mood History",
            data=csv_data,
            file_name="mood_history.csv",
            mime="text/csv",
            use_container_width=True
        )

        # -----------------------------
        # Clear History
        # -----------------------------

        if st.button(
            "🗑️ Clear Mood History",
            use_container_width=True
        ):

            st.session_state.history = []

            if os.path.exists(history_file):
                os.remove(history_file)

            st.rerun()

        st.subheader("📈 Mood Frequency")

        mood_counts = history_df["Mood"].value_counts()

        st.bar_chart(mood_counts)

    else:

        st.info(
            "No predictions yet. "
            "Go to Home and analyze some text."
        )

# ==================================================
# ABOUT MODEL PAGE
# ==================================================

elif page == "ℹ️ About Model":

    st.title("ℹ️ About AI Mood Detector")

    st.write(
        "This application uses Machine Learning to "
        "detect emotions expressed in text."
    )

    st.divider()

    # -----------------------------
    # Model Information
    # -----------------------------

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "The system uses Linear Support Vector Machine (SVM) "
        "for emotion classification."
    )

    st.subheader("🔢 Text Processing")

    st.write(
        "TF-IDF (Term Frequency-Inverse Document Frequency) "
        "is used to convert text into numerical features "
        "that the machine learning model can understand."
    )

    st.subheader("😊 Emotions Detected")

    emotions = [
        "😢 Sadness",
        "😊 Joy",
        "❤️ Love",
        "😡 Anger",
        "😨 Fear",
        "😲 Surprise"
    ]

    for emotion in emotions:
        st.write("•", emotion)

    st.subheader("📊 Model Accuracy")

    st.metric(
        "Test Accuracy",
        "88.60%"
    )

    st.subheader("⚙️ How the System Works")

    st.code(
        """
User enters text
       ↓
Text Cleaning
       ↓
TF-IDF Vectorization
       ↓
Linear SVM
       ↓
Emotion Prediction
       ↓
Confidence Score
        """
    )

    st.subheader("📚 Technologies Used")

    technologies = [
        "Python",
        "Pandas",
        "Scikit-learn",
        "TF-IDF",
        "Linear SVM",
        "Streamlit"
    ]

    for technology in technologies:
        st.write("•", technology)

    st.divider()

    st.warning(
        "⚠️ This application is an educational machine "
        "learning project. It detects emotions from text "
        "and should not be considered a medical or "
        "psychological diagnosis."
    )

    st.divider()