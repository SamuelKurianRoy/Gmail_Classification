import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
import seaborn as sns 
import matplotlib.pyplot as plt
import joblib
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
import schedule
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path

# Set up logging first
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level before loading env vars
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set environment file path
ENV_FILE_PATH = r'C:\Users\hai\Desktop\Youtube_Downloader_bot\Gmail_Enviornment_Variables.env'

# Load environment variables from specific path
if os.path.exists(ENV_FILE_PATH):
    load_dotenv(ENV_FILE_PATH)
    logger.info(f"Loaded environment variables from: {ENV_FILE_PATH}")
    # Update logging level after loading env vars
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))
else:
    logger.error(f"Environment file not found at: {ENV_FILE_PATH}")
    logger.info("Please ensure the environment file exists at the specified path")

# Email configuration for alerts
EMAIL_SENDER = os.getenv('EMAIL_SENDER')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
EMAIL_RECIPIENT = os.getenv('EMAIL_RECIPIENT')

# Model configuration
N_CLUSTERS = int(os.getenv('N_CLUSTERS', 3))
MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', 1))

# Dataset configuration
DATASET_PATH = os.getenv('DATASET_PATH')
DATASET_ENCODING = os.getenv('DATASET_ENCODING', 'ISO-8859-1')

def verify_config():
    """
    Verify that all required configuration is properly set up
    """
    required_vars = {
        'EMAIL_SENDER': EMAIL_SENDER,
        'EMAIL_PASSWORD': EMAIL_PASSWORD,
        'EMAIL_RECIPIENT': EMAIL_RECIPIENT,
        'DATASET_PATH': DATASET_PATH
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error("Configuration is incomplete. Please set the following environment variables in:")
        logger.error(f"Environment file: {ENV_FILE_PATH}")
        for var in missing_vars:
            logger.error(f"{var}: Required value not set")
        return False
    return True

def scrape_emails(url):
    """
    Scrape emails from a specified URL
    """
    try:
        logger.info(f"Attempting to scrape emails from: {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        emails = []
        for link in soup.find_all('a', href=True):
            if 'mailto:' in link['href']:
                emails.append(link['href'].replace('mailto:', ''))
        logger.info(f"Successfully scraped {len(emails)} emails")
        return emails
    except Exception as e:
        logger.error(f"Error scraping emails: {e}")
        return []

def send_alert(subject, message):
    """
    Send email alert
    """
    if not verify_config():
        logger.error("Cannot send alert: Configuration is incomplete")
        return False

    try:
        logger.info(f"Preparing to send alert: {subject}")
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = EMAIL_RECIPIENT
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        logger.info("Alert sent successfully!")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication Error: Please check your Gmail credentials")
        logger.error("Make sure you're using an App Password if you have 2FA enabled")
        logger.error("Visit: https://support.google.com/accounts/answer/185833 to create an App Password")
        return False
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
        return False

def perform_clustering(X_transformed, n_clusters=N_CLUSTERS):
    """
    Perform K-means clustering on the vectorized data
    """
    logger.info(f"Performing K-means clustering with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_transformed)
    logger.info("Clustering completed successfully")
    return clusters, kmeans

def monitor_and_alert():
    """
    Monitor new emails and send alerts for suspicious content
    """
    logger.info("Starting monitoring cycle")
    try:
        # Load saved models
        classifier = joblib.load('naive_bayes_Gmail_Classifier.pkl')
        vectorizer = joblib.load('tfidf_vectorizer.pkl')
        logger.info("Models loaded successfully")
        
        # Test with some sample emails
        test_emails = [
            "Congratulations! You've won a million dollars! Click here to claim your prize!",
            "Meeting reminder: Team sync at 2 PM tomorrow",
            "URGENT: Your account has been compromised! Click here to verify!"
        ]
        
        for email in test_emails:
            prediction = predict_email(email, classifier, vectorizer)
            logger.info(f"Email classified as: {prediction}")
            if prediction == "Spam":
                send_alert(
                    "Suspicious Email Detected",
                    f"A suspicious email was detected:\n\n{email}"
                )
    except Exception as e:
        logger.error(f"Error in monitoring cycle: {e}")

# Load dataset
logger.info("Loading dataset...")
df = pd.read_csv(DATASET_PATH, encoding=DATASET_ENCODING)

# Drop unnecessary columns
df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore', inplace=True)
df.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

# Data cleaning
df.drop_duplicates(inplace=True)
logger.info(f"Dataset loaded and cleaned. Shape: {df.shape}")

# Vectorize text data
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(df['Message'])

# Perform clustering
clusters, kmeans = perform_clustering(X_transformed)
df['Cluster'] = clusters

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Category'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train classifiers
logger.info("Training classifiers...")
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
logger.info("Classifiers trained successfully")

# Save models & vectorizer
joblib.dump(classifier, 'naive_bayes_Gmail_Classifier.pkl')
joblib.dump(log_reg, 'logistic_Gmail_Classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(kmeans, 'kmeans_clusterer.pkl')
logger.info("Models saved successfully")

# Function to predict new email
def predict_email(email_text, model, vectorizer):
    email_transformed = vectorizer.transform([email_text])
    prediction = model.predict(email_transformed)[0]
    return "Spam" if prediction == 1 else "Ham"

# Schedule monitoring
schedule.every(MONITORING_INTERVAL).hours.do(monitor_and_alert)

# Run the scheduler
if __name__ == "__main__":
    logger.info("Starting email monitoring system...")
    
    # Verify configuration before starting
    if not verify_config():
        logger.error("Please set up your configuration before running the script")
        logger.info(f"Update the environment variables in: {ENV_FILE_PATH}")
        logger.info("Required variables:")
        logger.info("EMAIL_SENDER=your.email@gmail.com")
        logger.info("EMAIL_PASSWORD=your_app_password")
        logger.info("EMAIL_RECIPIENT=recipient.email@example.com")
        logger.info("DATASET_PATH=path/to/your/dataset.csv")
        logger.info("DATASET_ENCODING=ISO-8859-1")
        logger.info("N_CLUSTERS=3")
        logger.info("MONITORING_INTERVAL=1")
        logger.info("LOG_LEVEL=INFO")
        exit(1)
    
    # Run initial monitoring cycle
    monitor_and_alert()
    
    while True:
        schedule.run_pending()
        time.sleep(60)


