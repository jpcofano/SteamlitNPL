import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import nltk
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from transformers import pipeline
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import torch
from transformers import BertTokenizerFast, EncoderDecoderModel
import spacy
from string import punctuation

nlp = spacy.load('es_core_news_sm')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
tokenizer = BertTokenizerFast.from_pretrained(ckpt)
model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

nltk.download('punkt')
nltk.download('vader_lexicon')
st.set_option('deprecation.showPyplotGlobalUse', False)

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

def analyze_sentiment(text):
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(text)
    sentiment_label = get_sentiment_label(sentiment_scores['compound'])
    return sentiment_label, sentiment_scores

def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positivo'
    elif score <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

def get_top_words(text, stop_words):
    words = text.lower().split()
    words = [word.strip(".,") for word in words if word.strip(".,") not in stop_words]
    word_counts = Counter(words)
    top_words = word_counts.most_common(5)
    return top_words

from nltk.tokenize import word_tokenize

def get_representative_phrase(text):
    soup = BeautifulSoup(text, 'html.parser')
    paragraphs = soup.find_all('p')
    sentences = [sentence.text.strip() for paragraph in paragraphs for sentence in sent_tokenize(paragraph.text)]

    if not sentences:
        return "No se encontraron oraciones representativas."
    else:
        words = [word.lower() for sentence in sentences for word in word_tokenize(sentence)]
        word_counts = Counter(words)
        top_words = word_counts.most_common(5)

        sentence_scores = []
        for sentence in sentences:
            score = sum([word_counts[word.lower()] for word in word_tokenize(sentence)])
            sentence_scores.append((sentence, score))

        ranked_sentences = sorted(sentence_scores, key=lambda x: (x[1], len(x[0])), reverse=True)

        if ranked_sentences:
            representative_phrase = ranked_sentences[0][0]
            return representative_phrase
        else:
            return "No se encontraron oraciones representativas."

def generate_word_cloud(text, stop_words):
    words = text.lower().split()
    words = [word.strip(".,") for word in words if word.strip(".,") not in stop_words]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    st.pyplot(plt.gcf())

def classify_sentences(text):
    doc = nlp(text)

    punctuation = punctuation + '\n'

    # Lista de stopwords en español
    stopwords = open('stopwords-es.txt', 'r', encoding='utf-8').read()

    # Buscar la frecuencia de las palabras
    word_frequencies = {}
    for word in doc:
        if word.text.lower() not in stopwords:
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    # Normalización de la frecuencia de palabras
    max_frequency = max(word_frequencies.values())
    for word in word_frequencies.keys():
        word_frequencies[word] = word_frequencies[word] / max_frequency

    # Tokenization de las oraciones
    sentence_tokens = [sent for sent in doc.sents]

    # Cálculo del puntaje de las oraciones
    sentence_scores = {}
    for sent in sentence_tokens:
        for word in sent:
            if word.text.lower() in word_frequencies.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent] += word_frequencies[word.text.lower()]

    return sentence_scores

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
                with open('stopwords-es.txt', 'r', encoding='utf-8') as file:
                    stopwords_es = file.read().splitlines()
                stop_words = set(stopwords.words('spanish')).union(stopwords_es)

                st.subheader("Análisis de Sentimiento")
                sentiment_label, sentiment_scores = analyze_sentiment(text_input)
                st.write("Puntuación de sentimiento:", sentiment_scores['compound'])
                st.image(get_sentiment_image(sentiment_label))

                st.subheader("Palabras más frecuentes")
                top_words = get_top_words(text_input, stop_words)
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
            generate_word_cloud(text_input, stop_words)

except Exception as e:
    st.error(f"Ocurrió un error: {str(e)}")
