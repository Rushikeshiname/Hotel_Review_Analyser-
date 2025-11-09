"""
Run with:
streamlit run streamlit_app.py
"""

import streamlit as st
from predict import SentimentAnalyzer

st.set_page_config(page_title="Review Analyser", layout="centered")

st.title("Review Analyser — DistilBERT")
st.write("Enter a review to get sentiment prediction (fine-tuned).")

model_dir = st.text_input("Model directory", value="./saved_distilbert")
if st.button("Load model"):
    with st.spinner("Loading model..."):
        analyzer = SentimentAnalyzer(model_dir)
    st.success("Model loaded. Enter text now.")
else:
    analyzer = None

text = st.text_area("Review text", height=120)
if st.button("Predict") and analyzer is not None and text.strip():
    with st.spinner("Predicting..."):
        out = analyzer.predict(text)
    st.write("**Label:**", out["label"])
    st.write("**Confidence:**", f"{out['score']:.3f}")
    st.write("**Probabilities:**", out["probs"])
elif st.button("Predict") and analyzer is None:
    st.error("Please load the model first (click 'Load model').")
