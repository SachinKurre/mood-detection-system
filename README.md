# Mood Detection System

A Machine Learning-based Mood Detection System that analyzes text input and predicts the emotion expressed in the text. The system uses **TF-IDF for text feature extraction** and **Support Vector Machine (SVM)** as the final classification model.

The project includes a **Streamlit web interface** that allows users to enter text and receive real-time emotion predictions.

## Features

- Text preprocessing and cleaning
- TF-IDF feature extraction
- SVM-based emotion classification
- Logistic Regression used for model comparison
- Real-time emotion prediction
- Interactive Streamlit web interface
- Prediction history
- Model evaluation

## Emotions Detected

The system detects six emotions:

- 😢 Sadness
- 😊 Joy
- ❤️ Love
- 😡 Anger
- 😨 Fear
- 😲 Surprise

## Machine Learning Model

### Support Vector Machine (SVM)

The **SVM model is the main and final model** used in this project for emotion classification.

The text data is first cleaned and transformed into numerical features using **TF-IDF (Term Frequency-Inverse Document Frequency)**. These features are then provided to the SVM classifier to predict the emotion expressed in the text.

### Model Comparison

**Logistic Regression** was trained as an initial model and used only for comparison with the SVM model.

After comparing the models, **SVM was selected as the final model** used in the Streamlit application.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Support Vector Machine (SVM)
- Logistic Regression
- Joblib
- Streamlit
- Matplotlib

## Project Workflow

```text
Text Input
    ↓
Text Cleaning & Preprocessing
    ↓
TF-IDF Feature Extraction
    ↓
SVM Classification Model
    ↓
Predicted Emotion
    ↓
Streamlit Web Interface
