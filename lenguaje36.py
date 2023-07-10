import streamlit as st
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import pandas as pd
import re
import base64
from io import BytesIO
from reportlab.pdfgen import canvas
from PIL import Image
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from reportlab.lib.pagesizes import letter
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np

nlp = spacy.load('es_core_news_sm')

# Función para obtener una lista de palabras vacías
def get_stopwords():
    with open('stopwords-es.txt', 'r', encoding='utf-8') as file:
        stopwords_list = file.read().split('\n')
    return stopwords_list

# Función para analizar la representatividad de las oraciones en un texto
def analyze_representativeness(sentences):
    keywords = set(get_keywords(' '.join([sentence.text for sentence in sentences])))
    scores = []
    for sentence in sentences:
        sentence_keywords = set(sentence.text.split())
        score = len(sentence_keywords.intersection(keywords)) / len(keywords)
        scores.append(score)
    return scores

# Función para imprimir las oraciones del resumen
def print_summary_sentences(text, summary_length, max_words=80):
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = analyze_representativeness(sentences)
    top_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:summary_length]

    # Limitar la longitud de las oraciones al máximo de palabras permitido
    summary_sentences = []
    for sentence, score in top_sentences:
        words = sentence.text.split()
        if len(words) > max_words:
            words = words[:max_words]
            # Agregar un punto al final si no existe
            if not re.search(r'[.!?]$', words[-1]):
                words[-1] += '.'
            sentence_text = ' '.join(words)
        else:
            sentence_text = sentence.text
        summary_sentences.append(sentence_text)

    return summary_sentences

# Función para generar un resumen extractivo del texto
def generate_extractive_summary(text, summary_length, max_words=40):
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = analyze_representativeness(sentences)
    top_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:summary_length]

    # Limitar la longitud de las oraciones al máximo de palabras permitido
    summary_sentences = []
    for sentence, score in top_sentences:
        words = sentence.text.split()
        if len(words) > max_words:
            words = words[:max_words]
            # Agregar un punto al final si no existe
            if not re.search(r'[.!?]$', words[-1]):
                words[-1] += '.'
            sentence_text = ' '.join(words)
        else:
            sentence_text = sentence.text
        summary_sentences.append(sentence_text)

    return ' '.join(summary_sentences)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# Resto del código...

# Estilo de página
st.set_page_config(
    page_title="Procesamiento de Texto",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
# Mostrar el logo
image = Image.open('logo-gcba.png')
st.image(image, width=800)
st.title("Procesamiento de Texto")

# Barra lateral
st.sidebar.title("Ingrese el texto")
text_input = st.sidebar.text_area("Ingrese el texto")
uploaded_files = st.sidebar.file_uploader("Cargar archivos .txt", type="txt", accept_multiple_files=True)

# Verificación del archivo cargado
if uploaded_files is not None:
    for file in uploaded_files:
        text_input += file.read().decode("utf-8")

# Separador
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Obtener enlace de descarga para el archivo PDF
def get_pdf_download_link(filename, content):
    packet = BytesIO()
    can = canvas.Canvas(packet)
    y = 720
    for line in content:
        if isinstance(line, tuple):  # Verificar si es una tupla
            line = line[0]  # Obtener el primer elemento de la tupla
        line_encoded = line.encode('utf-8')  # Codificar la cadena en UTF-8
        can.drawString(10, y, line_encoded)
        y -= 20
    can.save()

    packet.seek(0)
    b64_pdf = base64.b64encode(packet.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">Descargar PDF</a>'
    return href

# Análisis de Sentimiento
st.subheader("Análisis de Sentimiento")

# Tokenizar el texto en oraciones
sentences = sent_tokenize(text_input)

# Crear un objeto SentimentIntensityAnalyzer
sentiment = SentimentIntensityAnalyzer()

# Calcular la puntuación de sentimiento para cada oración
sentiment_scores = [sentiment.polarity_scores(sentence)["compound"] for sentence in sentences]

# Calcular la puntuación de sentimiento promedio
sentiment_total = np.mean(sentiment_scores)

# Mostrar la puntuación de sentimiento promedio
st.write(f"Puntuación de sentimiento: {sentiment_total}")

# Almacenar los resultados en la lista de contenido
content = []
content.append(("Análisis de Sentimiento", sentiment_total))

# Resto del código...

# Obtener el enlace de descarga para el PDF
pdf_link = get_pdf_download_link("resultados.pdf", content)
st.markdown(pdf_link, unsafe_allow_html=True)
