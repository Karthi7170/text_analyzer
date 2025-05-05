import streamlit as st
import plotly.express as px
import numpy as np
from wordcloud import WordCloud
from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize
from langdetect import detect
from deep_translator import GoogleTranslator

# Download necessary NLTK data
nltk.download('punkt', quiet=True)

# Set page configuration
st.set_page_config(
    page_title="Text Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load models
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    sentiment_analyzer = pipeline("sentiment-analysis")
    topic_classifier = pipeline("zero-shot-classification")
    title_generator = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    return summarizer, emotion_classifier, sentiment_analyzer, topic_classifier, title_generator

summarizer, emotion_classifier, sentiment_analyzer, topic_classifier, title_generator = load_models()

# Define functions
def detect_language(text):
    return detect(text)

def translate_to_english(text, src_lang):
    if src_lang != "en":
        return GoogleTranslator(source=src_lang, target="en").translate(text)
    return text

def generate_title(text):
    summary = title_generator(text, max_length=5, min_length=1, do_sample=False)
    return summary[0]['summary_text'].strip().split()[0]

def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def classify_emotion(text):
    results = emotion_classifier(text)
    return [(item['label'], item['score']) for item in results[0]]

def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

def classify_topic(text, candidate_labels=["Technology", "Health", "Finance", "Sports", "Politics", "Education","Entertainment"]):
    result = topic_classifier(text, candidate_labels)
    return result['labels'][0], result['scores'][0]

def generate_word_cloud(text):
    wordcloud = WordCloud(width=600, height=300, background_color='white').generate(text)
    st.image(wordcloud.to_array(), use_container_width=True)

def plot_emotion_chart(emotions):
    labels, scores = zip(*emotions)
    fig = px.bar(x=labels, y=scores, labels={'x': 'Emotions', 'y': 'Score'}, title="Emotion Analysis Results")
    st.plotly_chart(fig, use_container_width=True)

# UI layout
st.title("📝 Text Analyzer")

# Input box for text
user_input = st.text_area("Enter text for analysis:", height=100)

if st.button("Analyze"):
    if user_input:
        with st.spinner("Analyzing..."):
            detected_lang = detect_language(user_input)
            translated_text = translate_to_english(user_input, detected_lang)
            title = generate_title(translated_text)
            summary = summarize_text(translated_text)
            emotions = classify_emotion(translated_text)
            sentiment_label, sentiment_score = analyze_sentiment(translated_text)
            topic_label, topic_score = classify_topic(translated_text)
        st.success("Analysis complete!")
        
        st.subheader("🌍 Language Detection:")
        st.markdown(f"**Detected Language:** {detected_lang.upper()}")
        
        if detected_lang != "en":
            st.subheader("🔄 Translated Text:")
            st.write(translated_text)
        
        st.subheader("📌 Generated Title:")
        st.info(title)

        st.subheader("📊 Summary:")
        st.write(summary)
        
        st.subheader("😊 Emotion Analysis Result:")
        plot_emotion_chart(emotions)
        
        st.subheader("📈 Sentiment Analysis Result:")
        st.markdown(f"**Sentiment:** {sentiment_label} ({sentiment_score:.2%})")
        
        st.subheader("📌 Topic Classification:")
        st.markdown(f"**Topic:** {topic_label} ({topic_score:.2%})")
        
        st.subheader("🌟 Word Cloud Visualization:")
        generate_word_cloud(translated_text)
    else:
        st.warning("Please enter text for analysis.")
