import re
import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load spaCy with only the lemmatizer and tokenizer
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

# Customize stopwords if you want to preserve important domain terms
custom_stopwords = ENGLISH_STOP_WORDS - {"project", "team", "system", "design", "data",
    "process", "develop", "engineer", "application", "user",
    "performance", "solution", "support", "security", "testing"}

def clean_text(text):
    """
    Preprocess text by:
    - Lowercasing
    - Removing digits and punctuation
    - Lemmatizing
    - Removing stopwords and short tokens
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)                # Normalize whitespace
    text = re.sub(r'[^\w\s]', '', text)             # Remove punctuation
    doc = nlp(text)

    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in custom_stopwords
        and token.is_alpha
        and len(token) > 1
    ]

    return ' '.join(tokens)

def highlight_matches(text, resume_tokens):
    """
    Highlight words in job description that appear in the resume.
    """
    words = re.findall(r'\w+|\W+', text)
    highlighted = ""
    for word in words:
        if word.strip().lower() in resume_tokens:
            highlighted += f"<mark style='background-color: yellow; color: black;'>{word}</mark>"
        else:
            highlighted += word
    return highlighted

def highlight_resume_text(text, job_tokens):
    """
    Highlight words in resume that appear in the job description.
    """
    words = re.findall(r'\w+|\W+', text)
    highlighted = ""
    for word in words:
        if word.strip().lower() in job_tokens:
            highlighted += f"<mark style='background-color: lightgreen; color: black;'>{word}</mark>"
        else:
            highlighted += word
    return highlighted
