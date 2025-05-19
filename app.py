import streamlit as st
from transformers import BertForSequenceClassification, BertTokenizer
import torch
import re

# Basic Streamlit setup
st.set_page_config(layout="centered")
st.title("✉️ Email Tone Checker")

# Load model and tokenizer from results folder
@st.cache_resource
def load_model():
    model = BertForSequenceClassification.from_pretrained('./results')
    tokenizer = BertTokenizer.from_pretrained('./results')
    model.eval()  # Set to evaluation mode
    return model, tokenizer

model, tokenizer = load_model()

# Simple text cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9,.!?']", " ", text)
    return " ".join(text.split())

# Prediction function
def predict_tone(text):
    inputs = tokenizer(
        clean_text(text),
        return_tensors='pt',
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    tones = ['Neutral', 'Passive-Aggressive', 'Aggressive', 'Polite']
    return tones[outputs.logits.argmax().item()]

# App interface
email = st.text_input("Enter email text:")
if st.button("Check Tone"):
    if email:
        tone = predict_tone(email)
        st.write(f"**Tone:** {tone}")
    else:
        st.warning("Please enter some text")