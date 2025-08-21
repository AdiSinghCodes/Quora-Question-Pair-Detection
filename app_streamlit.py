

import xgboost as xgb
import pandas as pd
import numpy as np
import os
import re
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import distance
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle



# Load the trained XGBoost model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'xgb_model_350k.json')
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

# Load feature columns (assume same as training)
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'final_features.csv')
feature_columns = pd.read_csv(FEATURES_PATH, nrows=1).drop(['Unnamed: 0', 'id', 'is_duplicate'], axis=1).columns.tolist()

# Load frequency dictionaries (qid1, qid2)
try:
    with open(os.path.join(os.path.dirname(__file__), 'qid1_freq.pkl'), 'rb') as f:
        qid1_freq = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'qid2_freq.pkl'), 'rb') as f:
        qid2_freq = pickle.load(f)
except Exception:
    qid1_freq = {}
    qid2_freq = {}

# Load TFIDF vectorizer and fit on all questions (should be saved from training)
try:
    with open(os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl'), 'rb') as f:
        tfidf = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'word2tfidf.pkl'), 'rb') as f:
        word2tfidf = pickle.load(f)
except Exception:
    tfidf = None
    word2tfidf = {}

# Load spacy model
try:
    nlp = spacy.load('en_core_web_lg')
except Exception:
    nlp = spacy.load('en_core_web_md')

SAFE_DIV = 0.0001
STOP_WORDS = set(stopwords.words('english'))
porter = PorterStemmer()

# Contractions dictionary (shortened for brevity, add all as in your notebook)
contractions = {"can't": "can not", "won't": "will not", "n't": " not", "what's": "what is", "it's": "it is", "i'm": "i am", "'re": " are", "'s": " own", "'ll": " will"}

def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
         .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
         .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
         .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
         .replace("he's", "he is").replace("she's", "she is").replace("'s", " own")\
         .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
         .replace("€", " euro ").replace("'ll", " will")
    x = x.replace('@', ' at ')
    x = x.replace('[math]', '')
    x = x.replace(',000,000,000 ', 'b ')
    x = x.replace(',000,000 ', 'm ')
    x = x.replace(',000 ', 'k ')
    x = re.sub(r'([0-9]+)000000000', r'\1b', x)
    x = re.sub(r'([0-9]+)000000', r'\1m', x)
    x = re.sub(r'([0-9]+)000', r'\1k', x)
    x_tokens = []
    for word in x.split():
        if word in contractions:
            word = contractions[word]
        x_tokens.append(word)
    x = ' '.join(x_tokens)
    pattern = re.compile(r'\W')
    if isinstance(x, str):
        x = re.sub(pattern, ' ', x)
    if isinstance(x, str):
        x = porter.stem(x)
        example1 = BeautifulSoup(x, features="html.parser")
        x = example1.get_text()
    return x

def get_basic_features(q1, q2):
    q1len = len(q1)
    q2len = len(q2)
    q1_n_words = len(q1.split())
    q2_n_words = len(q2.split())
    q1_words_set = set(q1.split())
    q2_words_set = set(q2.split())
    word_Common = len(q1_words_set & q2_words_set)
    word_Total = len(q1_words_set) + len(q2_words_set)
    word_share = word_Common / (word_Total + SAFE_DIV)
    freq_q1_plus_q2 = q1len + q2len
    freq_q1_minus_q2 = abs(q1len - q2len)
    return [q1len, q2len, q1_n_words, q2_n_words, word_Common, word_Total, word_share, freq_q1_plus_q2, freq_q1_minus_q2]

def get_token_features(q1, q2):
    token_features = [0.0]*10
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features
    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])
    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])
    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))
    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features

def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)


def get_freq_feature(qid, freq_dict):
    try:
        return freq_dict[qid]
    except Exception:
        return 0

def get_w2v_features(text, word2tfidf, nlp, dim=384):
    doc = nlp(text)
    mean_vec = np.zeros((dim,))
    weight_sum = 0.0
    for word in doc:
        vec = word.vector
        idf = word2tfidf.get(str(word), 0)
        mean_vec += vec * idf
        weight_sum += idf
    if weight_sum > 0:
        mean_vec /= weight_sum
    return mean_vec

def preprocess_input(q1, q2, qid1=None, qid2=None):
    # Preprocess
    q1_clean = preprocess(q1)
    q2_clean = preprocess(q2)
    # Token features
    token = get_token_features(q1_clean, q2_clean)
    # Fuzzy features
    token_set_ratio = fuzz.token_set_ratio(q1_clean, q2_clean)
    token_sort_ratio = fuzz.token_sort_ratio(q1_clean, q2_clean)
    fuzz_ratio = fuzz.QRatio(q1_clean, q2_clean)
    fuzz_partial_ratio = fuzz.partial_ratio(q1_clean, q2_clean)
    # Longest substring ratio
    longest_substr_ratio = get_longest_substr_ratio(q1_clean, q2_clean)
    # Basic features
    q1len = len(q1_clean)
    q2len = len(q2_clean)
    q1_n_words = len(q1_clean.split())
    q2_n_words = len(q2_clean.split())
    q1_words_set = set(q1_clean.split())
    q2_words_set = set(q2_clean.split())
    word_Common = len(q1_words_set & q2_words_set)
    word_Total = len(q1_words_set) + len(q2_words_set)
    word_share = word_Common / (word_Total + SAFE_DIV)
    freq_q1_plus_q2 = q1len + q2len
    freq_q1_minus_q2 = abs(q1len - q2len)

    # Frequency features (qid1/qid2 must be provided or set to 0)
    freq_qid1 = get_freq_feature(qid1, qid1_freq) if qid1 is not None else 0
    freq_qid2 = get_freq_feature(qid2, qid2_freq) if qid2 is not None else 0

    # W2V features
    if nlp is not None and word2tfidf:
        w2v_q1 = get_w2v_features(q1_clean, word2tfidf, nlp)
        w2v_q2 = get_w2v_features(q2_clean, word2tfidf, nlp)
    else:
        w2v_q1 = np.zeros(384)
        w2v_q2 = np.zeros(384)

    # Compose feature vector in the same order as feature_columns
    feature_dict = {
        'cwc_min': token[0],
        'cwc_max': token[1],
        'csc_min': token[2],
        'csc_max': token[3],
        'ctc_min': token[4],
        'ctc_max': token[5],
        'last_word_eq': token[6],
        'first_word_eq': token[7],
        'abs_len_diff': token[8],
        'mean_len': token[9],
        'token_set_ratio': token_set_ratio,
        'token_sort_ratio': token_sort_ratio,
        'fuzz_ratio': fuzz_ratio,
        'fuzz_partial_ratio': fuzz_partial_ratio,
        'longest_substr_ratio': longest_substr_ratio,
        'freq_qid1': freq_qid1,
        'freq_qid2': freq_qid2,
        'q1len': q1len,
        'q2len': q2len,
        'q1_n_words': q1_n_words,
        'q2_n_words': q2_n_words,
        'word_Common': word_Common,
        'word_Total': word_Total,
        'word_share': word_share,
        'freq_q1+q2': freq_q1_plus_q2,
        'freq_q1-q2': freq_q1_minus_q2
    }

    # Add w2v features in correct order
    for i in range(384):
        feature_dict[f'{i}_x'] = w2v_q1[i]
    for i in range(384):
        feature_dict[f'{i}_y'] = w2v_q2[i]

    # Fill missing features with 0
    features = [feature_dict.get(col, 0) for col in feature_columns]
    features_df = pd.DataFrame([features], columns=feature_columns)
    return features_df


# --- Streamlit UI ---
st.set_page_config(page_title="Quora Duplicate Question Detector", page_icon="https://upload.wikimedia.org/wikipedia/commons/9/91/Quora_logo_2015.svg")

st.markdown("""
<div style="text-align:center;">
    <img src="https://upload.wikimedia.org/wikipedia/commons/9/91/Quora_logo_2015.svg" width="120"/>
    <img src="https://cdn-icons-png.flaticon.com/512/3062/3062634.png" width="60" style="margin-left:10px;opacity:0.92;"/>
</div>
""", unsafe_allow_html=True)

st.title("Quora Duplicate Question Detector")
st.write("Enter two questions to check if they are duplicates. Powered by XGBoost, spaCy, and NLP features.")

with st.form("quora_form"):
    q1 = st.text_area("Question 1", placeholder="Type your first question...")
    q2 = st.text_area("Question 2", placeholder="Type your second question...")
    submit = st.form_submit_button("Check Duplicate")

if submit:
    if q1.strip() and q2.strip():
        features = preprocess_input(q1, q2)
        dmatrix = xgb.DMatrix(features)
        prob = bst.predict(dmatrix)[0]
        label = int(prob > 0.5)
        result = "Duplicate" if label == 1 else "Not Duplicate"
        st.markdown(f"<div style='text-align:center; margin-top:30px;'><span style='font-size:1.5em; font-weight:bold; color:{'#2ecc71' if label==1 else '#e74c3c'};'>{result}</span><br><span style='font-size:1.1em;'>Probability: <b>{prob:.2%}</b></span></div>", unsafe_allow_html=True)
    else:
        st.warning("Please enter both questions.")

st.markdown("""
<div style="text-align:center; color:#888; font-size:15px; margin-top:32px;">
    Made with ❤️ for Quora question analysis &middot; 2025
</div>
""", unsafe_allow_html=True)
