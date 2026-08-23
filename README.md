# Mood Detection System

A Machine Learning-based Mood Detection System that analyzes text input and predicts the emotion expressed in the text. The system uses TF-IDF for text feature extraction and Support Vector Machine (SVM) as the final classification model.

The project also includes a Streamlit web interface for real-time emotion prediction.

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

- Sadness 😢
- Joy 😊
- Love ❤️
- Anger 😡
- Fear 😨
- Surprise 😲

## Machine Learning Model

### Support Vector Machine (SVM)

SVM is the main and final classification model used in this project.

The input text is first cleaned and preprocessed. TF-IDF is then used to convert the text into numerical features. These features are passed to the SVM classifier to predict the emotion.

### Model Comparison

Logistic Regression was trained as an initial model and used for comparison with the SVM model.

After comparing the models, SVM was selected as the final model used in the Streamlit application.

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
- Seaborn
- PyArrow

## Project Structure

```text
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
```

## Installation

Clone the repository:

git clone https://github.com/SachinKurre/mood-detection-system.git

cd mood-detection-system

## Install Required Libraries

pip install -r requirments.txt

## Run the Application

streamlit run app.py

The application will open in your web browser.

## Dataset

The project uses a text-based emotion dataset divided into training, testing, and validation datasets.

The dataset files are stored in Parquet format inside the dataset/ directory.

## Model Files

The final trained SVM model and its TF-IDF vectorizer are stored in the model/ directory:

- svm_mood_model.pkl
- svm_tfidf_vectorizer.pkl

These files are loaded by the Streamlit application to make predictions.

## Application

The Streamlit application provides an easy-to-use interface where users can enter text and receive the predicted emotion.

The application also maintains a history of previous predictions.

## Future Improvements

- Improve model performance
- Add more emotion categories
- Improve the user interface
- Add advanced mood analytics
- Deploy the application online
- Support multiple languages

## Author

Sachin Kurre

Artificial Intelligence and Machine Learning
