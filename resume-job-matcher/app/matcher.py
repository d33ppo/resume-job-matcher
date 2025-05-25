
# app/matcher.py
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class JobMatcher:
    def __init__(self, job_json_path, use_bert=False):
        # Load the job data from JSON
        with open(job_json_path, 'r') as file:
            job_data = json.load(file)

        # Convert each job to a flat format
        jobs = job_data['jobs']
        for job in jobs:
            # Flatten description and requirements
            job['description_text'] = ' '.join(job.get('description', []))
            job['requirement_text'] = ' '.join(job.get('requirements', []))
            # Combine both into a single text field for matching
            job['full_text'] = job['description_text'] + ' ' + job['requirement_text']

        # Create DataFrame
        self.df = pd.DataFrame(jobs)

        # Vectorize the combined job text
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.job_matrix = self.vectorizer.fit_transform(self.df['full_text'])

        self.use_bert = use_bert
        if use_bert:
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.bert_embeddings = self.bert_model.encode(self.df['full_text'].tolist(), convert_to_tensor=True)

    def match(self, resume_text, top_k=5):
        if self.use_bert:
            resume_vec = self.bert_model.encode([resume_text], convert_to_tensor=True)
            scores = cosine_similarity(resume_vec, self.bert_embeddings)[0]
        else:
            resume_vec = self.vectorizer.transform([resume_text])
            scores = cosine_similarity(resume_vec, self.job_matrix)[0]

        top_indices = scores.argsort()[-top_k:][::-1]
        # Return top job details
        return self.df.iloc[top_indices][['title', 'company', 'location', 'type', 'deadline', 'description', 'requirements', 'phone', 'url']]
    