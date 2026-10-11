import pickle
import re

import nltk
import streamlit as st


@st.cache_resource
def load():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    with open("model/lgbm.pickle", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf.pickle", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf, set(nltk.corpus.stopwords.words("english")), nltk.stem.WordNetLemmatizer()


model, tfidf, stop_words, lemmer = load()


def preprocess(text):
    # Same steps as preprocess.ipynb: lowercase, strip punctuation, drop stopwords, lemmatize
    text = re.sub(r"[^\w\s]+", "", text.lower())
    return " ".join(lemmer.lemmatize(w) for w in text.split() if w not in stop_words)


st.set_page_config(page_title="Suicide Text Detection")
st.title("Suicide Text Detection")
st.caption("LightGBM + char-level TF-IDF, trained on 232k Reddit posts (95% test accuracy).")

text = st.text_area("Enter an English text", height=150)

if st.button("Analyze", type="primary") and text.strip():
    proba = model.predict_proba(tfidf.transform([preprocess(text)]).toarray())[0]
    p_suicide = proba[list(model.classes_).index("suicide")]
    if p_suicide >= 0.5:
        st.error(f"Potential suicidal content ({p_suicide:.0%} confidence)")
    else:
        st.success(f"Non-suicidal ({1 - p_suicide:.0%} confidence)")

st.divider()
st.caption(
    "Research project, not a diagnostic tool. If you or someone you know is struggling, "
    "please reach out to a local crisis line (Indonesia: 119 ext. 8) or "
    "[findahelpline.com](https://findahelpline.com)."
)
