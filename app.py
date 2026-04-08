import streamlit as st
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
import distance
import nltk


# DOWNLOAD NLTK DATA
try:
    stopwords.words('english')
except:
    nltk.download('stopwords')


# LOAD MODELS (CACHED)
@st.cache_resource
def load_models():
    model = pickle.load(open('model.pkl', 'rb'))
    w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, w2v_model, st_model

model, w2v_model, model_st = load_models()


# PREPROCESS FUNCTION
def preprocess(text):
    return text.lower().strip()

# FEATURE FUNCTIONS
def test_common_words(q1, q2):
    return len(set(q1.split()).intersection(set(q2.split())))

def test_total_words(q1, q2):
    return len(set(q1.split())) + len(set(q2.split()))

def test_fetch_token_features(q1, q2):
    SAFE_DIV = 0.0001
    STOP_WORDS = stopwords.words("english")
    
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return [0.0]*8

    q1_words = set([w for w in q1_tokens if w not in STOP_WORDS])
    q2_words = set([w for w in q2_tokens if w not in STOP_WORDS])
    
    q1_stops = set([w for w in q1_tokens if w in STOP_WORDS])
    q2_stops = set([w for w in q2_tokens if w in STOP_WORDS])
    
    common_word = len(q1_words & q2_words)
    common_stop = len(q1_stops & q2_stops)
    common_token = len(set(q1_tokens) & set(q2_tokens))
    
    return [
        common_word/(min(len(q1_words), len(q2_words))+SAFE_DIV),
        common_word/(max(len(q1_words), len(q2_words))+SAFE_DIV),
        common_stop/(min(len(q1_stops), len(q2_stops))+SAFE_DIV),
        common_stop/(max(len(q1_stops), len(q2_stops))+SAFE_DIV),
        common_token/(min(len(q1_tokens), len(q2_tokens))+SAFE_DIV),
        common_token/(max(len(q1_tokens), len(q2_tokens))+SAFE_DIV),
        int(q1_tokens[-1] == q2_tokens[-1]),
        int(q1_tokens[0] == q2_tokens[0])
    ]

def test_fetch_length_features(q1, q2):
    q1_tokens = q1.split()
    q2_tokens = q2.split()
    
    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return [0.0]*3
    
    strs = list(distance.lcsubstrings(q1, q2))
    
    return [
        abs(len(q1_tokens) - len(q2_tokens)),
        (len(q1_tokens) + len(q2_tokens)) / 2,
        len(strs[0]) / (min(len(q1), len(q2)) + 1)
    ]

def test_fetch_fuzzy_features(q1, q2):
    return [
        fuzz.QRatio(q1, q2),
        fuzz.partial_ratio(q1, q2),
        fuzz.token_sort_ratio(q1, q2),
        fuzz.token_set_ratio(q1, q2)
    ]

def get_avg_w2v(sentence, model, vector_size=300):
    words = sentence.split()
    vectors = [model.wv[w] for w in words if w in model.wv]
    
    if len(vectors) == 0:
        return np.zeros(vector_size)
    
    return np.mean(vectors, axis=0)


# FEATURE CREATOR
def query_point_creator(q1, q2):
    q1 = preprocess(q1)
    q2 = preprocess(q2)
    
    features = []
    
    features += [len(q1), len(q2)]
    features += [len(q1.split()), len(q2.split())]
    
    cw = test_common_words(q1, q2)
    tw = test_total_words(q1, q2)
    
    features += [cw, tw, cw/tw if tw != 0 else 0]
    
    features += test_fetch_token_features(q1, q2)
    features += test_fetch_length_features(q1, q2)
    features += test_fetch_fuzzy_features(q1, q2)
    
    q1_w2v = get_avg_w2v(q1, w2v_model)
    q2_w2v = get_avg_w2v(q2, w2v_model)
    
    return np.hstack((
        np.array(features).reshape(1, 22),
        q1_w2v.reshape(1, -1),
        q2_w2v.reshape(1, -1)
    ))


# FINAL HYBRID PREDICTION
def final_prediction(q1, q2):
    
    # ML prediction
    features = query_point_creator(q1, q2)
    ml_pred = model.predict(features)[0]
    ml_prob = model.predict_proba(features)[0][1]
    
    # Semantic similarity
    embeddings = model_st.encode([q1, q2])
    similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]
    
    # Hybrid logic
    if similarity > 0.75:
        final = 1
    elif similarity < 0.5:
        final = 0
    else:
        final = ml_pred
    
    return final, ml_prob, similarity


# STREAMLIT UI
st.set_page_config(page_title="Duplicate Question Detector", layout="centered")

st.title("🔍 Duplicate Question Checker")
st.write("Check whether two questions are duplicates using AI + ML 🚀")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Check Duplicate"):
    if q1 and q2:
        
        result, prob, sim = final_prediction(q1, q2)
        
        if result == 1:
            st.success("✅ Duplicate Questions")
        else:
            st.error("❌ Not Duplicate Questions")
        
        # st.write(f"📊 ML Confidence: {prob:.2f}")
        # st.write(f"🧠 Semantic Similarity: {sim:.4f}")
        
        # st.progress(float(prob))
        
    else:
        st.warning("⚠️ Please enter both questions")