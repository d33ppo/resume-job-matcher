import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.utils import clean_text  # For lemmatization, stopword removal, etc.
import os
import joblib

class JobMatcher:
    def __init__(self, job_json_path, method="hybrid", alpha=0.5, model_name="all-MiniLM-L6-v2"):
        self.method = method.lower()
        self.alpha = alpha
        self.model_name = model_name

        self.df = self._load_jobs(job_json_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.vectorizer.fit_transform(self.df['full_text'])

        self.bert_model = SentenceTransformer(self.model_name)
        cache_path = "models/bert_embeddings.pkl"

        if os.path.exists(cache_path):
            self.bert_embeddings = joblib.load(cache_path)
        else:
            self.bert_embeddings = self.bert_model.encode(
                self.df['full_text'].tolist(),
                convert_to_tensor=True
            )
            joblib.dump(self.bert_embeddings, cache_path)

    def _load_jobs(self, path):
        with open(path, 'r') as file:
            job_data = json.load(file)

        jobs = job_data.get('jobs', [])
        for job in jobs:
            job['description_text'] = clean_text(' '.join(job.get('description', [])))
            job['requirement_text'] = clean_text(' '.join(job.get('requirements', [])))
            job['full_text'] = job['description_text'] + ' ' + job['requirement_text']

        return pd.DataFrame(jobs)

    def match(self, resume_text, top_k=5, return_columns=None):
        resume_text = clean_text(resume_text)

        tfidf_vec = self.vectorizer.transform([resume_text])
        tfidf_scores = cosine_similarity(tfidf_vec, self.job_matrix)[0]

        bert_vec = self.bert_model.encode([resume_text], convert_to_tensor=True)
        bert_scores = cosine_similarity(bert_vec, self.bert_embeddings)[0]

        if self.method == "tfidf":
            combined_scores = tfidf_scores
        elif self.method == "berth":
            combined_scores = bert_scores
        elif self.method == "hybrid":
            combined_scores = self.alpha * tfidf_scores + (1 - self.alpha) * bert_scores
        else:
            raise ValueError("Invalid method. Choose from 'tfidf', 'berth', or 'hybrid'.")

        top_indices = combined_scores.argsort()[-top_k:][::-1]
        top_matches = self.df.iloc[top_indices].copy()
        top_matches['match_score'] = combined_scores[top_indices]

        if return_columns is None:
            return_columns = [
                'title', 'company', 'location', 'type', 'deadline',
                'description', 'requirements', 'phone', 'url', 'match_score'
            ]

        return top_matches[[col for col in return_columns if col in top_matches.columns]]
