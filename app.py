import streamlit as st
import joblib
import re
from PIL import Image

# Load models and vectorizer
nb_model = joblib.load(r"C:\Users\hai\Desktop\Gmail_Classification\naive_bayes_Gmail_Classifier.pkl")
lr_model = joblib.load(r"C:\Users\hai\Desktop\Gmail_Classification\logistic_Gmail_Classifier.pkl")
vectorizer = joblib.load(r"C:\Users\hai\Desktop\Gmail_Classification\tfidf_vectorizer.pkl")

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
