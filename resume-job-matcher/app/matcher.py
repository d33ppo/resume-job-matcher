
# app/matcher.py
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class JobMatcher:
    def __init__(self, job_json_path):
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

    def match(self, resume_text, top_k=5):
        # Transform the resume into a vector
        resume_vec = self.vectorizer.transform([resume_text])

        # Compute cosine similarity between resume and all job vectors
        scores = cosine_similarity(resume_vec, self.job_matrix)[0]

        # Get top k job indices with highest similarity scores
        top_indices = scores.argsort()[-top_k:][::-1]

        # Return top job details
        return self.df.iloc[top_indices][['title', 'company', 'location', 'type', 'deadline', 'description', 'requirements', 'phone', 'url']]