import streamlit as st
import nltk
import torch
# from transformers import BartForConditionalGeneration, BartTokenizer
from nltk.sentiment import SentimentIntensityAnalyzer
# from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import pandas as pd
import re
import base64
# import os
from io import BytesIO
import requests
from reportlab.pdfgen import canvas
from PIL import Image
import spacy
# import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM
# from transformers import AutoTokenizer, AutoModelForCausalLM
from reportlab.lib.pagesizes import letter
from transformers import PegasusTokenizer
import sentencepiece as spm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import pyperclip
import re

nlp = spacy.load('es_core_news_sm')
# nlp = spacy.load('./es_core_news_sm-3.1.0')

if 'resultados' not in st.session_state:
    st.session_state.resultados = []

# def generate_extractive_summary(text):
#     doc = nlp(text)
#     summary = []
#     for sentence in doc.sents:
#         if len(sentence) <= 20:
#             summary.append(sentence.text)  # Extraer el texto del objeto Span
#     return ' '.join(summary)


def copiar_texto(texto):
    # Copiar el texto al portapapeles en Streamlit
    st.write(texto)
    st.button("Copiar al portapapeles", on_click=lambda: st.experimental_set_clipboard(texto))



# Resto del código...


def guardar_texto(texto):
    # Guardar el texto en un archivo de texto y descargarlo en Streamlit
    b64_texto = base64.b64encode(texto.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64_texto}" download="resumen.txt">Descargar resumen</a>'
    st.markdown(href, unsafe_allow_html=True)
# from nltk.corpus import stopwords

def get_stopwords():
    with open('stopwords-es.txt', 'r', encoding='utf-8') as file:
        stopwords_list = file.read().split('\n')
    return stopwords_list


def get_keywords(text):
    words = set(text.split())
    stop_words = get_stopwords()
    keywords = [word for word in words if word not in stop_words]
    return keywords

def analyze_representativeness(sentences):
    keywords = set(get_keywords(' '.join([sentence.text for sentence in sentences])))
    scores = []
    for sentence in sentences:
        sentence_keywords = set(sentence.text.split())
        score = len(sentence_keywords.intersection(keywords)) / len(keywords)
        scores.append(score)
    return scores

def print_summary_sentences(text, summary_length, max_words=40):
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = analyze_representativeness(sentences)
    top_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:summary_length]
    
    # Limitar la longitud de las sentencias al máximo de palabras permitido
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


def generate_extractive_summary(text, summary_length, max_words=40):
    doc = nlp(text)
    sentences = list(doc.sents)
    scores = analyze_representativeness(sentences)
    top_sentences = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)[:summary_length]
    
    # Limitar la longitud de las sentencias al máximo de palabras permitido
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




# text = 'This is a sentence about keywords. This is a sentence about topics. This is a sentence about both keywords and topics.'
# summary = generate_extractive_summary(text, 20)

# print(summary)


# Load the image using PIL
image = Image.open('logo-gcba.png')

# Display the image in Streamlit

# Logo URL
# logo_url = os.path.join(os.getcwd(), 'logo-gcba.png')


lexicon_path = "lexico_afinn_en_es.csv"

def get_sentiment_label(sentiment_total):
    if sentiment_total >= 0.05:
        return 'Positivo'
    elif sentiment_total <= -0.05:
        return 'Negativo'
    else:
        return 'Neutral'

def get_sentiment_image(sentiment_total):
    if sentiment_total >= 0.05:
        return 'positive.png'
    elif sentiment_total <= -0.05:
        return 'negative.png'
    else:
        return 'neutral.png'

def load_sentiment_lexicon():
    lexicon_path = 'lexico_afinn_en_es.csv'
    lexicon_df = pd.read_csv(lexicon_path, encoding='ISO-8859-1')
    return lexicon_df


def analyze_sentiment(text, language):
    sentiment_lexicon = load_sentiment_lexicon()
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(text)
    
    compound_score = sentiment_scores['compound']
    sentiment_label = get_sentiment_label(compound_score)
    
    if language == 'es':
        words = text.split()
        sentiment_scores['words_sentiment'] = [sentiment_lexicon.get(word.lower(), 0) for word in words]
    
    return sentiment_label, compound_score, sentiment_scores




# Lista para almacenar el contenido ejecutado
resultados = []

def guardar_pdf(content):
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
    pdf_filename = "resultados.pdf"
    with open(pdf_filename, "wb") as file:
        file.write(packet.getvalue())

    return pdf_filename



# Obtener enlace de descarga para el archivo PDF
def get_pdf_download_link(filename, text):
    with open(filename, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{text}</a>'
    return href


def get_image_download_link(img_bytes, filename, text):
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href



nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# def save_image(url):
#     response = requests.get(url, stream=True)
#     if response.status_code == 200:
#         with open("image.jpg", "wb") as file:
#             file.write(response.content)
#         st.success("Imagen guardada exitosamente.")

def save_image(imagen, nombre_archivo):
    # Guardar la imagen en BytesIO
    bytes_io = BytesIO()
    imagen.savefig(bytes_io, format='png')
    bytes_io.seek(0)

    # Crear el botón de descarga
    st.download_button(
        label='Descargar Imagen',
        data=bytes_io,
        file_name=nombre_archivo,
        mime='image/png'
    )
        
def remove_punctuation(text):
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text

# tokenizer = AutoTokenizer.from_pretrained("username/repo_name")
# model = AutoModel.from_pretrained("username/repo_name")

# from transformers import BertTokenizerFast, EncoderDecoderModel
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# ckpt = 'mrm8488/bert2bert_shared-spanish-finetuned-summarization'
# tokenizer = BertTokenizerFast.from_pretrained(ckpt)
# model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

# Load model directly

def generate_summary_bert(text, summary_length):
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-summarization")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/bert2bert_shared-spanish-finetuned-summarization")
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    output = model.generate(input_ids, max_length=summary_length, attention_mask=attention_mask)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def generate_summary_bart(text, summary_length):
    bart_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    bart_model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
    inputs = bart_tokenizer([text], max_length=1024, truncation=True, return_tensors='pt')
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=summary_length, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Load model directly


# BART model
def generate_summary_bart_spanish(text, summary_length):
    bart_tokenizer = AutoTokenizer.from_pretrained("mrm8488/bloom-560m-finetuned-wikilingua-spanish-summarization")
    bart_model = AutoModelForCausalLM.from_pretrained("mrm8488/bloom-560m-finetuned-wikilingua-spanish-summarization")
    inputs = bart_tokenizer([text], max_length=200, truncation=True, return_tensors='pt')
    summary_ids = bart_model.generate(inputs['input_ids'], num_beams=4, max_length=summary_length, early_stopping=True)
    summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def generate_summary_DeepESP_spanish(text, summary_length):
    num_beams = 4  # Número de secuencias generadas
    temperature = 0.8  # Controla la aleatoriedad de la generación (valores más altos = más aleatorio)
    top_k = 80  # Controla la diversidad de las palabras generadas (valores más altos = más diversidad)
    DeepESP_tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
    DeepESP_model = AutoModelForCausalLM.from_pretrained("DeepESP/gpt2-spanish")
    inputs = DeepESP_tokenizer([text], max_length=100, truncation=True, return_tensors='pt')
    summary_ids = DeepESP_model.generate(inputs['input_ids'],  num_beams=num_beams, max_length=summary_length, early_stopping=True,temperature=temperature, top_k=top_k)
    summary = DeepESP_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Marian model
# Load model directly

# marian_tokenizer = transformers.MarianTokenizer.from_pretrained('marian-base-es')
# marian_model = transformers.MarianMTModel.from_pretrained('marian-base-es').to(device)

def generate_summary_marian_spanish(text, summary_length):
    # Ajustar los parámetros de generación
    num_beams = 4  # Número de secuencias generadas
    temperature = 0.8  # Controla la aleatoriedad de la generación (valores más altos = más aleatorio)
    top_k = 50  # Controla la diversidad de las palabras generadas (valores más altos = más diversidad)
    marian_tokenizer = AutoTokenizer.from_pretrained("Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization")
    marian_model = AutoModelForSeq2SeqLM.from_pretrained("Narrativa/bsc_roberta2roberta_shared-spanish-finetuned-mlsum-summarization")
    inputs = marian_tokenizer([text], max_length=100, truncation=True, return_tensors='pt')
    summary_ids = marian_model.generate(inputs['input_ids'], num_beams=num_beams, max_length=summary_length, early_stopping=True,temperature=temperature, top_k=top_k)
    summary = marian_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary



def generate_summary_Pegasus(text, summary_length):
    model_name = "google/pegasus-xsum"
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
    batch = tokenizer([text], truncation=True, padding="longest", return_tensors="pt").to(device)
    translated = model.generate(**batch, max_length=summary_length)
    summary = tokenizer.batch_decode(translated, skip_special_tokens=True)
    if isinstance(summary, list) and len(summary) > 0:
        return summary[0]


# max_length=100
def generate_summary_NASES(text, summary_length):
    NASES_tokenizer = AutoTokenizer.from_pretrained("ELiRF/NASES")
    NASES_model = AutoModelForSeq2SeqLM.from_pretrained("ELiRF/NASES")
    inputs = NASES_tokenizer([text], max_length=514, truncation=True, return_tensors='pt')
    summary_ids = NASES_model.generate(inputs['input_ids'], num_beams=4, max_length=summary_length, early_stopping=True)
    summary = NASES_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# max_length=100
def generate_summary_flax(text, summary_length):
    num_beams = 4  # Número de secuencias generadas
    temperature = 0.8  # Controla la aleatoriedad de la generación (valores más altos = más aleatorio)
    top_k = 50  # Controla la diversidad de las palabras generadas (valores más altos = más diversidad)
    flax_tokenizer = AutoTokenizer.from_pretrained("flax-community/spanish-t5-small")
    flax_model = AutoModelForSeq2SeqLM.from_pretrained("flax-community/spanish-t5-small")
    inputs = flax_tokenizer([text], max_length=514, truncation=True, return_tensors='pt')
    summary_ids = flax_model.generate(inputs['input_ids'], num_beams=num_beams, max_length=summary_length, early_stopping=True,temperature=temperature, top_k=top_k)
    summary = flax_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def generar_resumen(summarizer, text_input, summary_length):
    try:
        if summarizer == "BART":
            summary = generate_summary_bart(text_input, summary_length)
        elif summarizer == "BERT":
            summary = generate_summary_bert(text_input, summary_length)
        elif summarizer == "BERT2":
            summary = generate_summary_bart_spanish(text_input, summary_length)
        elif summarizer == "DeepESP":
            summary = generate_summary_DeepESP_spanish(text_input, summary_length)
        elif summarizer == "marian":
            summary = generate_summary_marian_spanish(text_input, summary_length)
        elif summarizer == "Pegasus":
            summary = generate_summary_Pegasus(text_input, summary_length)
        elif summarizer == "flax":
            text_input = remove_punctuation(text_input)
            summary = generate_summary_flax(text_input, summary_length)
        elif summarizer == "gextractive":
            summary = generate_extractive_summary(text_input, summary_length)
        else:
            raise ValueError("Summarizer no válido.")
        
        # st.write(summary)
        resultados.append(("Generar Resumen", summary))       
        return summary
    except Exception as e:
        st.error(f"Se ha producido un error: {str(e)}")



def get_top_words(text, stop_words, num_words, min_word_length):
    words = nltk.word_tokenize(text.lower())
    words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation) not in stop_words and len(word) >= min_word_length]
    word_freq = nltk.FreqDist(words)
    top_words = word_freq.most_common(num_words)
    return top_words


def generate_wordcloud(text, max_words, stop_words, color_map, colores_fondo, min_word_length):
    stop_words = set(get_stopwords())
    words = text.lower().split()
    words = [word.strip(string.punctuation) for word in words if word.strip(string.punctuation) not in stop_words]
    if len(words) == 0:
        print("No hay palabras en el texto proporcionado.")
        return
    word_freq = nltk.FreqDist(words)
    wc = WordCloud(width=1000, height=500, background_color=colores_fondo, stopwords=stop_words, max_words=max_words, colormap=color_map).generate_from_frequencies(word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Nube de Palabras ({len(word_freq)} palabras)")
    plt.tight_layout()

    # Mostrar la imagen en Streamlit
    st.pyplot(plt.gcf())

    # Generar el enlace de descarga
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_bytes = img_buffer.getvalue()
    href = get_image_download_link(img_bytes, filename='wordcloud.png', text='Descargar WordCloud')
    st.markdown(href, unsafe_allow_html=True)
    return href


# Inicializar el estado
if 'resultados' not in st.session_state:
    st.session_state.resultados = []


# Estilo de página
st.set_page_config(
    page_title="Procesamiento de Texto",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
# Display logo
# st.image(logo_url, width=1500)
st.image(image, width=800)
st.title("Procesamiento de Texto")

# Barra lateral
st.sidebar.title("Ingrese el texto")
text_input = st.sidebar.text_area("Ingrese el texto")
# uploaded_file = st.sidebar.file_uploader("Cargar archivo .txt", type="txt")
uploaded_files = st.sidebar.file_uploader("Cargar archivos .txt", type="txt", accept_multiple_files=True)

# Verificación del archivo cargado
if uploaded_files is not None:
    for file in uploaded_files:
        text_input += file.read().decode("utf-8")
# Separador
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)


if st.sidebar.button("Guardar en PDF"):
    if st.button("Guardar PDF"):
        # Crear un objeto de lienzo PDF
        c = canvas.Canvas("output.pdf", pagesize=letter)

        # Guardar cada contenido en el archivo PDF
        for contenido in resultados:
            tipo_contenido = contenido[0]
            contenido_texto = contenido[1]

            # Guardar imágenes en el archivo PDF
            if isinstance(contenido_texto, str) and contenido_texto.startswith("<Figure"):
                figura = st.markdown_to_html(contenido_texto)
                c.showPage()
                c.save()

            # Guardar texto en el archivo PDF
            else:
                c.setFont("Helvetica", 12)
                c.drawString(100, 700, tipo_contenido)
                c.drawString(100, 680, contenido_texto)
                c.showPage()
                c.save()

# pdf_download_link = f'<a href="output.pdf" download>Descargar PDF</a>'
# st.sidebar.markdown(pdf_download_link, unsafe_allow_html=True)

# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Análisis de Sentimiento
st.subheader("Análisis de Sentimiento")
if st.button("Analizar"):
    sentences = sent_tokenize(text_input)
    sentiment_scores = []
    word_count = 0  # Variable para almacenar la cantidad total de palabras

    for sentence in sentences:
        sentiment_label, compound_score, scores = analyze_sentiment(sentence, language='es')
        sentiment_scores.append(scores)
        word_count += len(sentence.split())  # Contar las palabras de la oración y sumarlas al total

    sentiment_df = pd.DataFrame(sentiment_scores)
    sentiment_total = sentiment_df['compound'].sum() / word_count  # Dividir la puntuación por la cantidad de palabras

    # Resto del código...


    # sentiment_label, sentiment_scores = analyze_sentiment(text_input, language)
    col1, col2 = st.columns(2)
    col1.write(["Puntuación de sentimiento:", sentiment_total])
    col2.write(get_sentiment_label(sentiment_total))
    col2.image(get_sentiment_image(sentiment_total))
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].hist(sentiment_df['neg'], bins=10, color='red', alpha=0.5, edgecolor='black')
    ax[0, 0].set_title("Distribución de Sentimiento Negativo")
    ax[0, 0].set_xlabel("Sentimiento Negativo")
    ax[0, 0].set_ylabel("Frecuencia")

    ax[0, 1].hist(sentiment_df['neu'], bins=10, color='gray', alpha=0.5, edgecolor='black')
    ax[0, 1].set_title("Distribución de Sentimiento Neutral")
    ax[0, 1].set_xlabel("Sentimiento Neutral")
    ax[0, 1].set_ylabel("Frecuencia")

    ax[1, 0].hist(sentiment_df['pos'], bins=10, color='green', alpha=0.5, edgecolor='black')
    ax[1, 0].set_title("Distribución de Sentimiento Positivo")
    ax[1, 0].set_xlabel("Sentimiento Positivo")
    ax[1, 0].set_ylabel("Frecuencia")

    ax[1, 1].bar(['Compound'], [sentiment_total], color='purple', alpha=0.5,edgecolor='black')
    ax[1, 1].set_title("Puntuación de Sentimiento Compuesto")
    ax[1, 1].set_ylabel("Puntuación")

    plt.tight_layout()
    st.pyplot(fig)

    # Separador
    st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

    # Almacenar los resultados en el estado
    st.session_state.resultados.append(("Análisis de Sentimiento", sentiment_total))
    # Guardar y mostrar el botón de descarga del PDF
    content = [
        "Análisis de Sentimiento",
        "Puntuación de sentimiento: {}".format(sentiment_total),
        # Agrega aquí más información para incluir en el PDF
    ]
    pdf_filename = guardar_pdf(content)
    download_link = get_pdf_download_link(pdf_filename, "Descargar PDF")
    st.markdown(download_link, unsafe_allow_html=True)
# Mostrar los resultados generados en la interfaz
    # for resultado in st.session_state.resultados:
    #     st.write(resultado)

# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Palabras más frecuentes
st.subheader("Palabras más frecuentes")
# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

num_words = st.slider("Cantidad de Palabras", 5, 25, 10)
if st.button("Mostrar"):
    stop_words = get_stopwords()
    min_word_length = 3
    top_words = get_top_words(text_input, stop_words, num_words, min_word_length)
    if top_words:
        words, counts = zip(*top_words)
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts)
        plt.xlabel("Palabra")
        plt.ylabel("Frecuencia")
        plt.title("Palabras más frecuentes")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        save_image(plt, "palabras_frecuentes.png")
        resultados.append(("Palabras más frecuentes", top_words))
    else:
        st.warning("No se encontraron palabras frecuentes.")

# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Generar Resumen
st.subheader("Generar Resumen")

# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Crear la interfaz de Streamlit
# summarizer = st.selectbox("Seleccionar summarizer", ["BART", "BERT", "BERT2", "T5", "marian", "Pegasus", "flax", "gextractive"])
# text_input = st.text_area("Introducir texto")
summarizer_options = ["BART", "BERT", "BERT2", "DeepESP", "marian", "Pegasus", "flax", "gextractive"]

with st.expander("Seleccionar Modelo"):
    selected_option = st.radio("Modelo:", summarizer_options, index=summarizer_options.index("Pegasus"))

# Mostrar la selección total
summarizer = selected_option if selected_option else ""
st.write("Selección total:", summarizer)

summary_length = st.slider("Longitud del resumen", min_value=10, max_value=1000, value=250, step=5)

if st.button("Generar Resumen"):
    summary = generar_resumen(summarizer, text_input, summary_length)
    if summary:
        copiar_texto(summary)
        guardar_texto(summary)

# Separador
st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Nube de Palabras
st.subheader("Nube de Palabras")

# Separador
# st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

max_words = st.slider("Cantidad de Palabras en la Nube", 25, 300, 150)
cloud_type = st.radio("Tipo de Nube", ["wordcloud", "barchart"])
color_map = st.selectbox("Mapa de Colores", ["viridis", "plasma", "inferno", "magma", "cividis", "gray", "spring", "summer", "autumn", "winter", "cool", "Wistia", "hot", "coolwarm", "bwr", "seismic", "twilight", "afmhot", "rainbow", "jet"])
colores_fondo = st.selectbox("Color de Fondo", ['white', 'black', 'gray', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink'])
if st.button("Generar"):
    if cloud_type == "wordcloud":
        stop_words = get_stopwords()
        min_word_length=3
        top_words_cloud = get_top_words(text_input, stop_words, max_words, min_word_length)  # Obtener las palabras frecuentes
        if top_words_cloud:
            word_cloud_text = " ".join([word[0] for word in top_words_cloud])  # Unir las palabras en un solo texto
            generate_wordcloud(word_cloud_text, max_words, stop_words, color_map, colores_fondo, min_word_length=3)  # Pasar el texto a la función generate_wordcloud
            resultados.append(("Nube de Palabras", top_words_cloud))
        else:
            st.warning("No se encontraron palabras frecuentes.")
    else:
        stop_words = get_stopwords()
        min_word_length=3
        top_words_cloud = get_top_words(text_input, stop_words, max_words, min_word_length)
        if top_words_cloud:
            words_cloud, counts_cloud = zip(*top_words_cloud)
            plt.figure(figsize=(10, 5))
            plt.bar(words_cloud, counts_cloud)
            plt.xlabel("Palabra")
            plt.ylabel("Frecuencia")
            plt.title("Palabras más frecuentes (Nube de Palabras)")
            plt.xticks(rotation=45)
            plt.set_cmap(color_map)
            st.pyplot(plt.gcf())
            resultados.append(("Palabras más frecuentes (Nube de Palabras)", top_words_cloud))
        else:
            st.warning("No se encontraron palabras frecuentes.")

# Interfaz de Streamlit
st.title("Mejores Frases")

# Botón para mostrar las 10 mejores frases
summary_length = st.slider("Cantidad de Frases", 2, 10, 2)

if st.button("Frases"):
    frases = print_summary_sentences(text_input, summary_length)
    for i, frase in enumerate(frases, start=1):
        st.write(f"Frase {i}: {frase}")

    if frases:
        frases_texto = "\n".join(frases)  # Unir las frases en un solo texto separadas por saltos de línea
        st.button("Copiar al portapapeles", on_click=lambda: pyperclip.copy(frases_texto))
        guardar_texto(frases_texto)


# Mostrar los resultados generados en la interfaz
# for contenido in st.session_state.resultados:
#     st.write(contenido)

