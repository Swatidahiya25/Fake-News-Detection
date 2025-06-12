# app.py

import streamlit as st
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Streamlit UI
st.title("📰 Fake News Detection App")

st.write("Enter a news article below to check if it's Real or Fake.")

user_input = st.text_area("Enter News Text Here")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        result = "✅ Real News" if prediction == 1 else "🚫 Fake News"
        st.success(f"Prediction: {result}")


