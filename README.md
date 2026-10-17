# Roman Urdu Sentiment Analyzer

Streamlit app that classifies Roman Urdu text as positive / negative / neutral
(TF-IDF word + character n-grams + Logistic Regression).

## Run
```
pip install -r requirements.txt
streamlit run app.py
```

## Features
- Single text analysis with confidence chart
- Bulk CSV analysis (column: text) with downloadable results
- Upload your own training dataset (columns: text,label) from the sidebar
