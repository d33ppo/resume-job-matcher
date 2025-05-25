# app/utils.py
import re

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def highlight_matches(text, resume_tokens):
    """
    Highlight words in `text` that are also in `resume_tokens` by changing the background color.
    """
    words = re.findall(r'\w+|\W+', text)  # Splits into words and punctuation
    highlighted = ""
    for word in words:
        if word.strip().lower() in resume_tokens:
            # Use HTML <mark> to highlight with a background color
            highlighted += f"<mark style='background-color: yellow;'>{word}</mark>"
        else:
            highlighted += word
    return highlighted

def highlight_resume_text(text, job_tokens):
    words = re.findall(r'\w+|\W+', text)
    highlighted = ""
    for word in words:
        if word.strip().lower() in job_tokens:
            highlighted += f"<mark style='background-color: lightgreen;'>{word}</mark>"
        else:
            highlighted += word
    return highlighted
