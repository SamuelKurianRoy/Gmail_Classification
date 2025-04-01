import streamlit as st
import joblib
import re
from PIL import Image
import requests

def download_file(url, filename):
    """Download a file from a given URL and save it locally."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download {filename}. HTTP Status Code: {response.status_code}")

# Define file URLs (Use the raw content link from GitHub)
base_url = "https://raw.githubusercontent.com/SamuelKurianRoy/Gmail_Classification/master/"
files = {
    "nb_model.pkl": "logistic_Gmail_Classifier.pkl",
    "lr_model.pkl": "naive_bayes_Gmail_Classifier.pkl",
    "vectorizer.pkl": "tfidf_vectorizer.pkl"
}

# Download and load models
for local_name, file_name in files.items():
    download_file(base_url + file_name, local_name)

# Load models and vectorizer
nb_model = joblib.load("nb_model.pkl")
lr_model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("Models and vectorizer loaded successfully!")


# Set page config
st.set_page_config(page_title="Spam Detector", page_icon="📧", layout="centered")

# Custom CSS for styling
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #222;
            font-weight: bold;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            font-size: 18px;
            background-color: #4A4A4A;
            color: white;
        }
        .stTextArea>div>textarea {
            border-radius: 10px;
            font-size: 16px;
            background-color: white;
            color: black;
        }
        .not-spam {
            color: black !important;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for model selection
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Function to set selected model
def set_model(model_name):
    st.session_state.selected_model = model_name

# Header section
st.markdown("""<h1 style='text-align: center;'>📧 Email Spam Detector</h1>""", unsafe_allow_html=True)
st.write("### Select a model and enter text to detect spam!")

# Model selection buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("🤖 Naïve Bayes"):
        set_model("nb")
with col2:
    if st.button("⚡ Logistic Regression"):
        set_model("lr")

# User input
user_input = st.text_area("✍️ Enter your email text:", height=150)

# Predict function
def predict(text):
    if not text:
        st.warning("⚠️ Please enter some text to predict.")
        return None
    
    if st.session_state.selected_model is None:
        st.warning("⚠️ Please select a model first.")
        return None

    text_tfidf = vectorizer.transform([text])

    if st.session_state.selected_model == "nb":
        prediction = nb_model.predict(text_tfidf)[0]
    elif st.session_state.selected_model == "lr":
        prediction = lr_model.predict(text_tfidf)[0]

    return "🚀 Spam" if prediction == 1 else "✅ <span class='not-spam'>Not Spam</span>"

# Predict button
if st.button("🔍 Predict"):
    result = predict(user_input)
    if result:
        st.markdown(f"<div class='not-spam'>{result}</div>", unsafe_allow_html=True)
