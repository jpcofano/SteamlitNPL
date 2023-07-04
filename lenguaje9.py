import streamlit as st
from langdetect import detect
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel


#import spacy
#from string import punctuation as punct

# nlp = spacy.load('es_core_news_sm')


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

nltk.download('punkt')
nltk.download('vader_lexicon')
st.set_option('deprecation.showPyplotGlobalUse', False)

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from string import punctuation


def get_representative_phrase(text, max_length=70):
    sentences = sent_tokenize(text)
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stopwords.words('english') and word not in punctuation]
    word_freq = FreqDist(words)
    sentence_scores = []

    for sentence in sentences:
        sentence_words = [word.lower() for word in word_tokenize(sentence)]
        score = sum(word_freq[word] for word in sentence_words)
        sentence_scores.append(score)

    max_score_index = sentence_scores.index(max(sentence_scores))
    representative_phrase = sentences[max_score_index]

    # Resumir la frase a una longitud máxima específica
    if len(representative_phrase) > max_length:
        representative_phrase = representative_phrase[:max_length] + "..."

    return representative_phrase


def get_sentiment_image(sentiment_label):
    if sentiment_label == 'Positivo':
        return 'positive.png'
    elif sentiment_label == 'Negativo':
        return 'negative.png'
    else:
        return 'neutral.png'

def generate_summary(text):
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=200, min_length=40, num_beams=4, repetition_penalty=2.0, length_penalty=1.0) 
    return tokenizer.decode(output[0], skip_special_tokens=True)

class WordSearch:
    def __init__(self, text):
        self.text = text
        self.word_scores = Counter(text.split())

    def generate(self):
        """Generate a word soup from the text."""
        word_list = []
        for word, count in self.word_scores.items():
            for _ in range(count):
                word_list.append(word)
        return "".join(sorted(word_list))

def detect_language(text):
    return detect(text)

def analyze_sentiment(text, language):
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(text)
    sentiment_label = get_sentiment_label(sentiment_scores['compound'], language)
    return sentiment_label, sentiment_scores

def get_sentiment_label(score, language):
    if language == 'es':
        if score >= 0.05:
            return 'Positivo'
        elif score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutro'
    elif language == 'en':
        if score >= 0.05:
            return 'Positive'
        elif score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

def get_top_words(text, language, stop_words):
    words = text.lower().split()
    words = [word.strip(".,") for word in words if word.strip(".,") not in stop_words]
    word_counts = Counter(words)
    top_words = word_counts.most_common(15)
    return top_words

def generate_word_cloud(text, language, stop_words):
    words = text.lower().split()
    words = [word.strip(".,") for word in words if word.strip(".,") not in stop_words]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt.gcf())



try:
    st.set_page_config(page_title="Análisis de Texto", layout="wide")
    st.title("Análisis de Texto")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Texto de Entrada")
        text_input = st.text_area("Ingrese el texto para analizar")

        if st.button("Analizar"):
            if text_input.strip() == "":
                st.warning("Por favor, ingrese un texto para analizar.")
            else:
                st.subheader("Idioma")
                language = detect_language(text_input)
                st.write(language)

                with open('stopwords-es.txt', 'r', encoding='utf-8') as file:
                    stopwords_es = file.read().splitlines()
                stop_words = set(stopwords.words('spanish')).union(stopwords_es)

                st.subheader("Análisis de Sentimiento")
                sentiment_label, sentiment_scores = analyze_sentiment(text_input, language)
                st.write("Puntuación de sentimiento:", sentiment_scores['compound'])
                st.image(get_sentiment_image(sentiment_label))

                st.subheader("Palabras más frecuentes")
                top_words = get_top_words(text_input, language, stop_words)
                word_freq_string = "  ".join([f"{word} ({count})" for word, count in top_words])
                st.write(word_freq_string)

                st.subheader("Frase Representativa")
                representative_phrase = get_representative_phrase(text_input)
                st.write(representative_phrase)

                st.subheader("Resumen")
                summary = generate_summary(text_input)
                st.write(summary)

    with col2:
        if text_input.strip() != "":
            st.subheader("Nube de Palabras")
            generate_word_cloud(text_input, language, stop_words)

except Exception as e:
    st.error(f"Ocurrió un error: {str(e)}")

