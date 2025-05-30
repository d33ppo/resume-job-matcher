import pandas as pd
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class JobMatcher:
    def __init__(self, job_json_path, use_bert=True, alpha=0.5, model_name="all-MiniLM-L6-v2"):
        """
        Initializes the JobMatcher.
        
        Parameters:
        - job_json_path (str): Path to JSON file containing job postings.
        - use_bert (bool): Whether to enable semantic embedding using SBERT.
        - alpha (float): Weight for TF-IDF in score fusion. 0 = only BERT, 1 = only TF-IDF.
        - model_name (str): SentenceTransformer model to use.
        """
        self.use_bert = use_bert
        self.alpha = alpha
        self.model_name = model_name

        # Load and flatten job postings
        self.df = self._load_jobs(job_json_path)

        # Vectorize job text using TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.vectorizer.fit_transform(self.df['full_text'])

        # Embed jobs using SBERT (if enabled)
        if self.use_bert:
            self.bert_model = SentenceTransformer(self.model_name)
            self.bert_embeddings = self.bert_model.encode(
                self.df['full_text'].tolist(),
                convert_to_tensor=True
            )

    def _load_jobs(self, path):
        """
        Load and flatten job JSON.
        Returns a DataFrame.
        """
        with open(path, 'r') as file:
            job_data = json.load(file)

        jobs = job_data.get('jobs', [])
        for job in jobs:
            job['description_text'] = ' '.join(job.get('description', []))
            job['requirement_text'] = ' '.join(job.get('requirements', []))
            job['full_text'] = job['description_text'] + ' ' + job['requirement_text']

        return pd.DataFrame(jobs)

    def match(self, resume_text, top_k=5, return_columns=None):
        """
        Matches the resume to job descriptions and returns top-k matches.

        Parameters:
        - resume_text (str): Preprocessed resume text
        - top_k (int): Number of top matches to return
        - return_columns (list): Optional list of columns to return
        
        Returns:
        - DataFrame of top-k matching jobs with scores
        """
        # TF-IDF similarity
        tfidf_vec = self.vectorizer.transform([resume_text])
        tfidf_scores = cosine_similarity(tfidf_vec, self.job_matrix)[0]

        # BERT similarity (if enabled)
        if self.use_bert:
            bert_vec = self.bert_model.encode([resume_text], convert_to_tensor=True)
            bert_scores = cosine_similarity(bert_vec, self.bert_embeddings)[0]
        else:
            bert_scores = np.zeros_like(tfidf_scores)

        # Combine scores
        combined_scores = self.alpha * tfidf_scores + (1 - self.alpha) * bert_scores
        top_indices = combined_scores.argsort()[-top_k:][::-1]

        top_matches = self.df.iloc[top_indices].copy()
        top_matches['match_score'] = combined_scores[top_indices]

        # Default columns to return
        if return_columns is None:
            return_columns = [
                'title', 'company', 'location', 'type', 'deadline',
                'description', 'requirements', 'phone', 'url', 'match_score'
            ]

        return top_matches[[col for col in return_columns if col in top_matches.columns]]

    