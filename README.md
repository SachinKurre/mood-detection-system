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

##Project Structure


mood-detection-system/
│
├── app.py
├── compare_models.py
├── download_dataset.py
├── evaluate_svm.py
├── predict.py
├── train.py
├── train_svm.py
├── requirment.txt
│
├── data/
│   └── mood_history.csv
│
├── dataset/
│   ├── train.parquet
│   ├── test.parquet
│   └── validation.parquet
│
└── model/
    ├── mood_model.pkl
    ├── tfidf_vectorizer.pkl
    ├── svm_mood_model.pkl
    ├── svm_tfidf_vectorizer.pkl
    └── confusion_matrix.png
    ├── tfidf_vectorizer.pkl
    ├── svm_mood_model.pkl
    ├── svm_tfidf_vectorizer.pkl
    └── confusion_matrix.png

##Installation

Clone the repository:

git clone https://github.com/YourUsername/mood-detection-system.git
cd mood-detection-system

##Install the required libraries:

pip install -r requirment.txt

##Run the Application

Run the Streamlit application:

streamlit run app.py

The application will open in your web browser.

##Dataset

The project uses a text-based emotion dataset divided into training, testing, and validation datasets.

The dataset files are stored in Parquet format inside the dataset/ directory.

##Model Files

The trained SVM model and its TF-IDF vectorizer are stored in the model/ directory:

svm_mood_model.pkl
svm_tfidf_vectorizer.pkl

These files are loaded by the Streamlit application to make predictions.

##Application

The Streamlit application provides an easy-to-use interface where users can enter a sentence or text and receive the predicted emotion.

The application also maintains a history of previous predictions.

##Future Improvements
Improve model performance
Add more emotion categories
Improve the user interface
Add advanced mood analytics
Deploy the application online
Support multiple languages

##Author

Sachin Kurre

Artificial Intelligence and Machine Learning
