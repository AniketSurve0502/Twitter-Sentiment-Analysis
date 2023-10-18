import streamlit as st
import joblib
import re

from sklearn.feature_extraction.text import CountVectorizer

# Load the trained model and CountVectorizer
model = joblib.load('trained_model.pkl')
bow_vectorizer = joblib.load('countt_vectorizer.pkl')

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = ' '.join(text.split())
    return text.lower()

def predict_sentiment(input_text):
    preprocessed_input = preprocess_text(input_text)
    preprocessed_input_bow = bow_vectorizer.transform([preprocessed_input])
    prediction = model.predict(preprocessed_input_bow)
    return "Racist/Sexist Statement" if prediction == 1 else "Non-Racist/Non-Sexist Statement"

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter any tweet and this system will identify whether it is racist/sexist tweet or not!!!")
user_input = st.text_input("Enter a sentence:")
if st.button("Predict"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Given sentence is : {sentiment}")
