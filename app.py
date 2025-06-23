import streamlit as st
from textblob import TextBlob

text = "I really enjoy using this app. It's fantastic!"
blob = TextBlob(text)
print(blob.sentiment)

st.title("Sentiment Analyzer")

text = st.text_area("Enter your text here")

if st.button("Analyze"):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0:
        st.success("Sentiment: Positive 😊")
    elif polarity < 0:
        st.error("Sentiment: Negative 😞")
    else:
        st.warning("Sentiment: Neutral 😐")
