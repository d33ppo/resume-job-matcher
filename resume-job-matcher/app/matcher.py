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
    def __init__(self, job_json_path, use_bert=True, alpha=0.5, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the JobMatcher with support for hybrid similarity (TF-IDF + BERT).

        Args:
            job_json_path (str): Path to job postings in JSON format.
            use_bert (bool): Enable BERT for semantic similarity.
            alpha (float): Weight for TF-IDF in the final similarity score.
            model_name (str): SentenceTransformer model to use.
        """
        self.use_bert = use_bert
        self.alpha = alpha
        self.model_name = model_name

        # Load and clean job descriptions into a DataFrame
        self.df = self._load_jobs(job_json_path)

        # Build TF-IDF vector representation
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.vectorizer.fit_transform(self.df['full_text'])

        # Initialize BERT model and optionally load or compute job embeddings
        if self.use_bert:
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
        """
        Loads and preprocesses job data: flattens and cleans description + requirements.

        Args:
            path (str): JSON file path.

        Returns:
            pd.DataFrame: Flattened and cleaned job data.
        """
        with open(path, 'r') as file:
            job_data = json.load(file)

        jobs = job_data.get('jobs', [])

        for job in jobs:
            # Clean and combine description and requirements into a single searchable string
            job['description_text'] = clean_text(' '.join(job.get('description', [])))
            job['requirement_text'] = clean_text(' '.join(job.get('requirements', [])))
            job['full_text'] = job['description_text'] + ' ' + job['requirement_text']

        return pd.DataFrame(jobs)

    def match(self, resume_text, top_k=5, return_columns=None):
        """
        Matches resume text to job descriptions using TF-IDF and optionally BERT.

        Args:
            resume_text (str): Raw resume text (will be cleaned internally).
            top_k (int): Number of best job matches to return.
            return_columns (list): Optional list of columns to return.

        Returns:
            pd.DataFrame: Top-k job matches with match_score column.
        """
        # Preprocess resume text
        resume_text = clean_text(resume_text)

        # Compute TF-IDF similarity
        tfidf_vec = self.vectorizer.transform([resume_text])
        tfidf_scores = cosine_similarity(tfidf_vec, self.job_matrix)[0]

        # Compute BERT similarity (if enabled)
        if self.use_bert:
            bert_vec = self.bert_model.encode([resume_text], convert_to_tensor=True)
            bert_scores = cosine_similarity(bert_vec, self.bert_embeddings)[0]
        else:
            bert_scores = np.zeros_like(tfidf_scores)

        # Combine both scores using weighted average
        combined_scores = self.alpha * tfidf_scores + (1 - self.alpha) * bert_scores

        # Select top-k matches based on combined score
        top_indices = combined_scores.argsort()[-top_k:][::-1]
        top_matches = self.df.iloc[top_indices].copy()
        top_matches['match_score'] = combined_scores[top_indices]

        # Return specified or default set of columns
        if return_columns is None:
            return_columns = [
                'title', 'company', 'location', 'type', 'deadline',
                'description', 'requirements', 'phone', 'url', 'match_score'
            ]

        return top_matches[[col for col in return_columns if col in top_matches.columns]]
