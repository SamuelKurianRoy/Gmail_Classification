import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib

# Load dataset
df = pd.read_csv(r'C:\Users\hai\Downloads\gmail.csv', encoding='ISO-8859-1')

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore', inplace=True)
df.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

# Data cleaning
df.drop_duplicates(inplace=True)

# Vectorize text data
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(df['Message'])

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Category'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train classifiers
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Save models & vectorizer
joblib.dump(classifier, 'naive_bayes_Gmail_Classifier.pkl')
joblib.dump(log_reg, 'logistic_Gmail_Classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict new email
def predict_email(email_text, model, vectorizer):
    email_transformed = vectorizer.transform([email_text])
    prediction = model.predict(email_transformed)[0]
    return "Spam" if prediction == 1 else "Ham"


