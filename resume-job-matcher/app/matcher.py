# app/matcher.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class JobMatcher:
    def __init__(self, job_csv_path):
        self.df = pd.read_csv(job_csv_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.vectorizer.fit_transform(self.df['description'])

    def match(self, resume_text, top_k=5):
        resume_vec = self.vectorizer.transform([resume_text])
        scores = cosine_similarity(resume_vec, self.job_matrix)[0]
        top_indices = scores.argsort()[-top_k:][::-1]
        return self.df.iloc[top_indices][['job_title', 'description']]
