# Roman Urdu Sentiment Analyzer - single file version
# Run: pip install streamlit scikit-learn pandas
#      streamlit run streamlit_app.py
import io
import re

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

st.set_page_config(page_title="Roman Urdu Sentiment", page_icon="💬")

SAMPLE = """text,label
yeh product bohat acha hai,positive
delivery time par aa gayi mazza agaya,positive
quality zabardast hai paisa wasool,positive
mujhe bohat pasand aya,positive
shandar service thi,positive
behtareen cheez hai sab ko recommend karunga,positive
customer support ne foran help ki,positive
packing bohat achi thi,positive
kamal ka phone hai battery lajawab,positive
value for money hai love it,positive
bohat khoobsurat design hai,positive
seller ka behavior bohat achha tha,positive
bakwas product hai paisa zaya,negative
delivery bohat late hui,negative
quality bohat kharab hai,negative
fraud seller hai kabhi mat kharidna,negative
phone do din mein kharab ho gaya,negative
worst service thi,negative
packing toot gayi thi,negative
customer support ne koi jawab nahi diya,negative
bilkul faltu cheez hai,negative
rang tasveer se mukhtalif hai,negative
mehnga hai lekin quality ghatiya,negative
bohat bura experience raha,negative
order abhi tak nahi aya,neutral
size medium hai,neutral
kal delivery hogi,neutral
colour black hai,neutral
price 2000 rupay hai,neutral
product box mein aya,neutral
average sa hai na acha na bura,neutral
theek thaak hai,neutral
kaam chal jata hai,neutral
warranty ek saal ki hai,neutral
mene kal order kiya tha,neutral
"""

EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}


def normalize(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # achaaaa -> achaa
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource(show_spinner="Model train ho raha hai...")
def train(csv_text: str):
    df = pd.read_csv(io.StringIO(csv_text)).dropna(subset=["text", "label"])
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2))),
        ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb",
                                 ngram_range=(2, 5))),
    ])
    model = Pipeline([
        ("features", features),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return model.fit(df["text"], df["label"]), len(df)


st.title("💬 Roman Urdu Sentiment Analyzer")
st.caption("Roman Urdu text likho, model positive / negative / neutral batayega.")

with st.sidebar:
    st.header("Training data")
    train_file = st.file_uploader("Apna dataset (CSV: text,label)", type="csv")
    csv_text = train_file.getvalue().decode("utf-8") if train_file else SAMPLE

model, n_rows = train(csv_text)
st.sidebar.write(f"Trained on **{n_rows}** rows")

tab1, tab2 = st.tabs(["Single text", "CSV upload"])

with tab1:
    text = st.text_area("Review / tweet",
                        "yeh product bohat acha hai, delivery bhi fast thi")
    if st.button("Analyze", type="primary") and text.strip():
        probs = model.predict_proba([text])[0]
        label = model.classes_[probs.argmax()]
        st.subheader(f"{EMOJI.get(label, '')} {label.upper()}")
        st.bar_chart(pd.Series(probs, index=model.classes_, name="confidence"))

with tab2:
    file = st.file_uploader("CSV (column name: text)", type="csv", key="bulk")
    if file:
        df = pd.read_csv(file)
        if "text" not in df.columns:
            st.error("CSV mein 'text' column hona chahiye.")
        else:
            df["sentiment"] = model.predict(df["text"].astype(str))
            st.dataframe(df)
            st.bar_chart(df["sentiment"].value_counts())
            st.download_button("Download results", df.to_csv(index=False),
                               "results.csv", "text/csv")
